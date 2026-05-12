# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: On 12h timeframe, enter long when price breaks above Camarilla R1 level with 1d uptrend (EMA34) and volume >2x average; enter short when price breaks below S1 level with 1d downtrend and volume >2x average. Exit when price reverses to Camarilla pivot point or trend changes. Designed for low trade frequency (<30/year) to minimize fee drag while capturing institutional levels in both bull and bear markets.
# Camarilla levels are widely watched by institutions, providing high-probability breakout/breakdown levels.
# Volume confirmation filters out false breakouts. Trend filter ensures alignment with higher timeframe momentum.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values

    # Calculate Camarilla levels from previous day
    # Camarilla formula: 
    # R4 = close + (high-low)*1.5/2
    # R3 = close + (high-low)*1.25/2
    # R2 = close + (high-low)*1.166/2
    # R1 = close + (high-low)*1.0833/2
    # PP = (high+low+close)/3
    # S1 = close - (high-low)*1.0833/2
    # S2 = close - (high-low)*1.166/2
    # S3 = close - (high-low)*1.25/2
    # S4 = close - (high-low)*1.5/2
    # We use previous day's OHLC to calculate today's levels
    
    # Calculate daily ranges and close
    daily_range = high_1d - low_1d
    camarilla_multiplier = 1.0833 / 2  # For R1 and S1
    
    # Shift to get previous day's values (lookback by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_range = daily_range * np.roll(np.ones_like(high_1d), 1)  # previous day's range
    prev_range = prev_high - prev_low
    
    # Handle first element
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_range[0] = high_1d[0] - low_1d[0]
    
    # Calculate Camarilla R1 and S1 from previous day
    r1 = prev_close + prev_range * camarilla_multiplier
    s1 = prev_close - prev_range * camarilla_multiplier
    pp = (prev_high + prev_low + prev_close) / 3  # Pivot point for exit
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 2x 24-period average (2 days of 12h bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 1d uptrend + volume spike
            if (close[i] > r1_aligned[i-1] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + 1d downtrend + volume spike
            elif (close[i] < s1_aligned[i-1] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point OR trend turns down
            if close[i] <= pp_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point OR trend turns up
            if close[i] >= pp_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals