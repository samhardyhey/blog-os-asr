# coding: utf-8

"""
    Speech Services API v3.0

    Speech Services API v3.0.  # noqa: E501

    OpenAPI spec version: v3.0
    
    Generated by: https://github.com/swagger-api/swagger-codegen.git
"""


import pprint
import re  # noqa: F401

import six

from swagger_client.configuration import Configuration


class DatasetProperties(object):
    """NOTE: This class is auto generated by the swagger code generator program.

    Do not edit the class manually.
    """

    """
    Attributes:
      swagger_types (dict): The key is attribute name
                            and the value is attribute type.
      attribute_map (dict): The key is attribute name
                            and the value is json key in definition.
    """
    swagger_types = {
        "accepted_line_count": "int",
        "rejected_line_count": "int",
        "duration": "str",
        "email": "str",
        "error": "EntityError",
    }

    attribute_map = {
        "accepted_line_count": "acceptedLineCount",
        "rejected_line_count": "rejectedLineCount",
        "duration": "duration",
        "email": "email",
        "error": "error",
    }

    def __init__(
        self,
        accepted_line_count=None,
        rejected_line_count=None,
        duration=None,
        email=None,
        error=None,
        _configuration=None,
    ):  # noqa: E501
        """DatasetProperties - a model defined in Swagger"""  # noqa: E501
        if _configuration is None:
            _configuration = Configuration()
        self._configuration = _configuration

        self._accepted_line_count = None
        self._rejected_line_count = None
        self._duration = None
        self._email = None
        self._error = None
        self.discriminator = None

        if accepted_line_count is not None:
            self.accepted_line_count = accepted_line_count
        if rejected_line_count is not None:
            self.rejected_line_count = rejected_line_count
        if duration is not None:
            self.duration = duration
        if email is not None:
            self.email = email
        if error is not None:
            self.error = error

    @property
    def accepted_line_count(self):
        """Gets the accepted_line_count of this DatasetProperties.  # noqa: E501

        The number of lines accepted for this data set.  # noqa: E501

        :return: The accepted_line_count of this DatasetProperties.  # noqa: E501
        :rtype: int
        """
        return self._accepted_line_count

    @accepted_line_count.setter
    def accepted_line_count(self, accepted_line_count):
        """Sets the accepted_line_count of this DatasetProperties.

        The number of lines accepted for this data set.  # noqa: E501

        :param accepted_line_count: The accepted_line_count of this DatasetProperties.  # noqa: E501
        :type: int
        """

        self._accepted_line_count = accepted_line_count

    @property
    def rejected_line_count(self):
        """Gets the rejected_line_count of this DatasetProperties.  # noqa: E501

        The number of lines rejected for this data set.  # noqa: E501

        :return: The rejected_line_count of this DatasetProperties.  # noqa: E501
        :rtype: int
        """
        return self._rejected_line_count

    @rejected_line_count.setter
    def rejected_line_count(self, rejected_line_count):
        """Sets the rejected_line_count of this DatasetProperties.

        The number of lines rejected for this data set.  # noqa: E501

        :param rejected_line_count: The rejected_line_count of this DatasetProperties.  # noqa: E501
        :type: int
        """

        self._rejected_line_count = rejected_line_count

    @property
    def duration(self):
        """Gets the duration of this DatasetProperties.  # noqa: E501

        The total duration of the datasets if it contains audio files. The duration is encoded as ISO 8601 duration  (\"PnYnMnDTnHnMnS\", see https://en.wikipedia.org/wiki/ISO_8601#Durations).  # noqa: E501

        :return: The duration of this DatasetProperties.  # noqa: E501
        :rtype: str
        """
        return self._duration

    @duration.setter
    def duration(self, duration):
        """Sets the duration of this DatasetProperties.

        The total duration of the datasets if it contains audio files. The duration is encoded as ISO 8601 duration  (\"PnYnMnDTnHnMnS\", see https://en.wikipedia.org/wiki/ISO_8601#Durations).  # noqa: E501

        :param duration: The duration of this DatasetProperties.  # noqa: E501
        :type: str
        """

        self._duration = duration

    @property
    def email(self):
        """Gets the email of this DatasetProperties.  # noqa: E501

        The email address to send email notifications to in case the operation completes.  The value will be removed after successfully sending the email.  # noqa: E501

        :return: The email of this DatasetProperties.  # noqa: E501
        :rtype: str
        """
        return self._email

    @email.setter
    def email(self, email):
        """Sets the email of this DatasetProperties.

        The email address to send email notifications to in case the operation completes.  The value will be removed after successfully sending the email.  # noqa: E501

        :param email: The email of this DatasetProperties.  # noqa: E501
        :type: str
        """

        self._email = email

    @property
    def error(self):
        """Gets the error of this DatasetProperties.  # noqa: E501


        :return: The error of this DatasetProperties.  # noqa: E501
        :rtype: EntityError
        """
        return self._error

    @error.setter
    def error(self, error):
        """Sets the error of this DatasetProperties.


        :param error: The error of this DatasetProperties.  # noqa: E501
        :type: EntityError
        """

        self._error = error

    def to_dict(self):
        """Returns the model properties as a dict"""
        result = {}

        for attr, _ in six.iteritems(self.swagger_types):
            value = getattr(self, attr)
            if isinstance(value, list):
                result[attr] = list(
                    map(lambda x: x.to_dict() if hasattr(x, "to_dict") else x, value)
                )
            elif hasattr(value, "to_dict"):
                result[attr] = value.to_dict()
            elif isinstance(value, dict):
                result[attr] = dict(
                    map(
                        lambda item: (item[0], item[1].to_dict())
                        if hasattr(item[1], "to_dict")
                        else item,
                        value.items(),
                    )
                )
            else:
                result[attr] = value
        if issubclass(DatasetProperties, dict):
            for key, value in self.items():
                result[key] = value

        return result

    def to_str(self):
        """Returns the string representation of the model"""
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        """For `print` and `pprint`"""
        return self.to_str()

    def __eq__(self, other):
        """Returns true if both objects are equal"""
        if not isinstance(other, DatasetProperties):
            return False

        return self.to_dict() == other.to_dict()

    def __ne__(self, other):
        """Returns true if both objects are not equal"""
        if not isinstance(other, DatasetProperties):
            return True

        return self.to_dict() != other.to_dict()
