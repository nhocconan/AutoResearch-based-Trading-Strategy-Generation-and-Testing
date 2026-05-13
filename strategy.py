#!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    P = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * P - low_1d
    S1 = 2 * P - high_1d
    R2 = P + (high_1d - low_1d)
    S2 = P - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (use previous day's levels)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)

    # Volume confirmation: current volume > 2.0 x 30-period average (stricter filter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(P_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R2 with volume spike and 1d EMA uptrend
            if close[i] > R2_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S2 with volume spike and 1d EMA downtrend
            elif close[i] < S2_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to daily pivot point (P)
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to daily pivot point (P)
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
 #!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    P = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * P - low_1d
    S1 = 2 * P - high_1d
    R2 = P + (high_1d - low_1d)
    S2 = P - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (use previous day's levels)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)

    # Volume confirmation: current volume > 2.0 x 30-period average (stricter filter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(P_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R2 with volume spike and 1d EMA uptrend
            if close[i] > R2_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S2 with volume spike and 1d EMA downtrend
            elif close[i] < S2_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to daily pivot point (P)
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to daily pivot point (P)
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
#!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    P = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * P - low_1d
    S1 = 2 * P - high_1d
    R2 = P + (high_1d - low_1d)
    S2 = P - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (use previous day's levels)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)

    # Volume confirmation: current volume > 2.0 x 30-period average (stricter filter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(P_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R2 with volume spike and 1d EMA uptrend
            if close[i] > R2_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S2 with volume spike and 1d EMA downtrend
            elif close[i] < S2_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to daily pivot point (P)
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to daily pivot point (P)
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
#!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    P = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * P - low_1d
    S1 = 2 * P - high_1d
    R2 = P + (high_1d - low_1d)
    S2 = P - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (use previous day's levels)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)

    # Volume confirmation: current volume > 2.0 x 30-period average (stricter filter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(P_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R2 with volume spike and 1d EMA uptrend
            if close[i] > R2_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S2 with volume spike and 1d EMA downtrend
            elif close[i] < S2_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to daily pivot point (P)
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to daily pivot point (P)
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
#!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    P = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * P - low_1d
    S1 = 2 * P - high_1d
    R2 = P + (high_1d - low_1d)
    S2 = P - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (use previous day's levels)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)

    # Volume confirmation: current volume > 2.0 x 30-period average (stricter filter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(P_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R2 with volume spike and 1d EMA uptrend
            if close[i] > R2_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S2 with volume spike and 1d EMA downtrend
            elif close[i] < S2_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to daily pivot point (P)
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to daily pivot point (P)
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
#!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    P = (high_1d + low_1d + close_1d) / 3.0
    R1 = 2 * P - low_1d
    S1 = 2 * P - high_1d
    R2 = P + (high_1d - low_1d)
    S2 = P - (high_1d - low_1d)

    # Align pivot levels to 4h timeframe (use previous day's levels)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)

    # Volume confirmation: current volume > 2.0 x 30-period average (stricter filter)
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(P_aligned[i]) or np.isnan(S2_aligned[i]) or 
            np.isnan(R2_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R2 with volume spike and 1d EMA uptrend
            if close[i] > R2_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: break below S2 with volume spike and 1d EMA downtrend
            elif close[i] < S2_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to daily pivot point (P)
            if close[i] <= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: price returns to daily pivot point (P)
            if close[i] >= P_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals
#!/usr/bin/env python3
# 4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike
# Hypothesis: Use daily pivot points (R2, S2) as stronger support/resistance than R1/S1.
# Enter long when price breaks above R2 with volume spike and 1d EMA50 uptrend.
# Enter short when price breaks below S2 with volume spike and 1d EMA50 downtrend.
# Exit when price returns to the daily pivot point (P).
# R2/S2 breakouts are less frequent but higher quality, reducing trade frequency and improving edge.
# Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 15-25 trades/year per symbol.

name = "4h_PivotPoint_3Level_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 1d data for pivot points and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate daily pivot points: P, R1, S1, R2, S2
    # P = (H + L + C) / 3
    # R1 = P + (R1 - S1) where R1 = 2*P - L, S1 = 2*P - H
    #