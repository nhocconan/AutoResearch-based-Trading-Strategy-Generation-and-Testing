# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for previous close and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Previous day's close (used as breakout level and exit level)
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan

    # Align previous close to 12h timeframe
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(prev_close_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above previous close with volume spike and 1d EMA uptrend
            if close[i] > prev_close_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below previous close with volume spike and 1d EMA downtrend
            elif close[i] < prev_close_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below previous close or revert to previous close
            if close[i] < prev_close_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            elif close[i] <= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above previous close or revert to previous close
            if close[i] > prev_close_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            elif close[i] >= prev_close_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals

#!/usr/bin/env python3
# 12h_MidnightBreakout_1dTrend
# Hypothesis: Capture daily momentum by trading breakouts at market open (00:00 UTC).
# Enter long when price breaks above the previous day's close with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below the previous day's close with volume confirmation and 1d EMA50 downtrend.
# Exit on opposite breakout or when price reverts to the previous day's close.
# Uses 12h timeframe with 1d trend filter to capture daily momentum with low trade frequency.
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-30 trades/year per symbol.

name = "12h_MidnightBreakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd