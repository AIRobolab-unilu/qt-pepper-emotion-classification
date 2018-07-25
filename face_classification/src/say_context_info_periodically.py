#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
import numpy as np

class ContextReporter:

	def __init__(self):
		# Initialize the node with rosp
		rospy.init_node('context_vocal_reporter_node', anonymous=True)
		self.speech_publisher = rospy.Publisher("/test/qt_tts/say", String,queue_size=1)
		self.image_sub = rospy.Subscriber("/qt_face/setEmotion", String, self.vocal_report_callback)
		rospy.loginfo("node initialized")



	def vocal_report_callback(self,data):
		report_msg = String()
		report_msg.data = 'It looks like you are ' + data.data + '. Is that true?'
		self.speech_publisher.publish(report_msg)





if __name__ == '__main__':
	print "................................................"
	context_reporter = ContextReporter()
	while not rospy.is_shutdown():
		try:
			rospy.sleep(15.)
			rospy.spin()
		except KeyboardInterrupt:
			print("Shutting down")
