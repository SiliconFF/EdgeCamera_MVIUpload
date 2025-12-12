<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload">
    <img src="images/logo.png" alt="Logo" width="200" height="200">
  </a>

<h3 align="center">MVI-Dragonfly</h3>

  <p align="center">
    Turn any computer into a lightweight edge camera that can upload directly to an MVI Edge device 
    eliminating time synchronization issues that can occur with large networks and RTSP streams.
    <br />
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload">View Demo</a>
    &middot;
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



## Disclaimer
*MVI is a trademark of International Business Machines Corporation.*

MVI-Dragonfly is an independent open-source project and is not affiliated with, endorsed by, or supported by IBM.



<!-- ABOUT THE PROJECT -->
## About The Project

<!--![Product Name Screen Shot][product-screenshot]](https://example.com)-->

For some backstory on this project. I have been implementing applications in assembly processes that use IBM MVI Edge to run quality inspection and object detection models. 
The single biggest hurdle that was encountered was the synchronization of the time of trigger and the time of the RTSP frame that was captured. To more clearly layout the issue 
that was encountered please see the chart below:

<img width="751" height="433" alt="rtspjitter" src="https://github.com/user-attachments/assets/634871c1-31af-4742-a2d5-4156c6700218" />  
<br><br/>
This chart shows the jitter of time difference between the rtsp timeline and the realworld timeline which when triggering MVI-Edge will cause there to be a discrepency. While there are tools in the MVI-Edge capture settings 
such as visual trigger and trigger orchestrations these are mostly bandaids for the greater issue.
<br><br/>
This issue gave me the idea to implement an interface to upload photos taken on an edge device (like a raspberrypi) connected to the camera to "Image Folder" devices on the MVI Edge. With this implmentation the compute overhead in the MVI server is reduced and has been tested to vastly improve the time sync from sometimes being off by upwards of 4s to being within ~200ms. The main design goal is to give the integrators of MVI Edge more flexibility with what cameras they can use and to deal with networking and financial restrictions on equipment usage.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

* [![Python][Python]][Python-url]
* OpenCV
* Paho MQTT
* Requests
* PyYAML

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Quick Start
If you just want to go straight to running the script this is just the steps to deploy
```bash
git clone https://github.com/SiliconFF/EdgeCamera_MVIUpload.git
cd EdgeCamera_MVIUpload
python -m venv venv && source venv/bin/activate   # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
cp sample_camera_edge_config.yaml camera_edge_config.yaml
# edit config to match your settings and include the certificates in the working directory if required
python uploader.py
```

<!-- GETTING STARTED -->
## Getting Started

Below are the steps needed to deploy and run MVI-Dragonfly with any device


### Prerequisites

This program is built entirely on Python3.9 and it is recommended that you create a virtual environment

  ```sh
  python -m venv [your_virtual_env_name]
  ```
**Then activate the virtual environment:**

  _Windows_
  ```sh
  .\[your_virtual_env_name]\Scripts\activate
  ```
  
  _Linux/RPI_
  ```sh
  source ./[your_virtual_env_name]/bin/activate
  ```



### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/SiliconFF/EdgeCamera_MVIUpload.git
   ```
2. Navigate to the new directory
   ```sh
   cd ./EdgeCamera_MVIUpload
   ```
3. Install the required packages
   ```sh
   pip install -r requirements.txt
   ```
4. Configure your camera_edge_config.yaml (this file must be named identically and in the same directory as uploader.py)    

  There is a sample yaml available ([sample_camera_edge_config.yaml](https://github.com/SiliconFF/EdgeCamera_MVIUpload/blob/main/sample_camera_edge_config.yaml))
  
  ```yaml
  #MVI Config
  mvi-edge-endpoint: "ExampleDomain:443/api/v1" #MVI Edge Endpoint
  mvi-username: "your_username"
  mvi-password: "your_password"
  mvi-device-uuid: "your_target_MVI_device_UUID" #Device UUID found in MVI Portal
  
  
  #MQTT Config
  mqtt-broker: "BrokerDomain" #Could be the same as MVI edge domain if you are using on board broker
  mqtt-port: 8883 #Use 8883 for TLS, 1883 for non-TLS
  mqtt-tls-required: True
  mqtt-tls-file-name: "yourcertificatefile.crt"
  mqtt-trigger-topic: "Your/Trigger/Topic"
  
  #Camera Config
  camera-type: "USB" #RTSP, USB, or PICAM
  camera-ip: "" #Must Include the rtsp:// prefix for RTSP cameras and the full stream path
  camera-width: "1920" 
  camera-height: "1080"
  
  #General
  host-platform: "WINDOWS" #Valid options are WINDOWS, LINUX, or RPI
  ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Once you have completed the prerequisites and installation steps you simply have to plug in your camera and with your virtual env active run the following in the root directory:

```sh
python .\uploader.py
```

The system will immediately validate your config settings and attempt to connect to your MVI-Edge, MQTT broker, and camera. Operational logs can be found in the automatically generated `uploader.log` in the root directory

**_It is highly recommended to set the script to run on startup which the steps vary depending on your environment_**



### Example Scenario:

RTSP streaming in MVI Edge exhibits poor time synchronization between the photo capture request and the actual frame capture in real time. This issue is particularly problematic for time-sensitive applications, such as processes that wait for inspection completion or involve moving objects.
To address this, consider one of the following alternatives:

Connect the RTSP camera directly to a single-board computer (e.g., Raspberry Pi) or use a different camera interface, such as CSI or USB. (There are plans to add GigE support though this seems overkill for this application)

These options allow connection to your MVI Edge instance with continuous frame capture. The system will wait for a message on a designated MQTT trigger topic of your choice. Upon receiving the message, a frame is captured and uploaded to your specified image folder for inspection.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap
- [ ] Verify functionality with RPI CSI modules
- [ ] Allow multiple devices to be set as a target
    - [ ] Devices can be linked to different trigger MQTT topics for more comprehensive coverage   
- [ ] Develop a supervisor webserver for management and config changes
- [ ] Support for reading from GigE cameras
- [ ] Prebuilt RPI install script to automatically configure a raspberry pi to be deployed


See the [open issues](https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgements

Thanks to IBM for the Maximo Visual Inspection Edge API that powers this project. [Access your MVI-Edge Swagger](https://www.ibm.com/docs/en/masv-and-l/maximo-vi/cd?topic=o-rest-apis) Page to learn more
This project is unofficial and not endorsed by IBM.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<a href="https://github.com/SiliconFF/EdgeCamera_MVIUpload/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SiliconFF/EdgeCamera_MVIUpload" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the MIT License. See [`LICENSE`][license-url] for more information.


<!-- CONTACT -->
## Contact
Collin Finetti - finetticj@gmail.com

Project Link: [https://github.com/SiliconFF/EdgeCamera_MVIUpload](https://github.com/SiliconFF/EdgeCamera_MVIUpload)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[contributors-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[forks-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/network/members
[stars-shield]: https://img.shields.io/github/stars/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[stars-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/stargazers
[issues-shield]: https://img.shields.io/github/issues/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[issues-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/issues
[license-shield]: https://img.shields.io/github/license/SiliconFF/EdgeCamera_MVIUpload.svg?style=for-the-badge
[license-url]: https://github.com/SiliconFF/EdgeCamera_MVIUpload/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/collin-finetti
[product-screenshot]: images/project_logo.png

<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[Python-url]: https://www.python.org/
