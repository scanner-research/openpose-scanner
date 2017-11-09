from scannerpy import Database, Job, BulkJob, DeviceType
import numpy as np

with Database() as db:
    db.load_op('build/libopenpose_op.so')

    # db.ingest_videos([('example', 'example.mp4')])

    frame = db.ops.FrameInput()
    frame_sampled = frame.sample()
    pose = db.ops.OpenPose(frame = frame_sampled, device=DeviceType.GPU)
    output = db.ops.Output(columns=[pose])
    job = Job(op_args={
        frame: db.table('example').column('frame'),
        frame_sampled: db.sampler.gather(range(10)),
        output: 'example_pose'
    })
    bulk_job = BulkJob(output=output, jobs=[job])
    [output] = db.run(bulk_job, force=True)

    for i, buf in output.column('pose').load():
        if len(buf) == 1: continue
        kp = np.frombuffer(buf, dtype=np.float32)
        print(kp)[:6]
