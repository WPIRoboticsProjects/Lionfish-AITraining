from pytube import YouTube

YouTube('https://www.youtube.com/watch?v=JxSPWOxYu7Y').streams.first().download()