# 101011
# 6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses daily trend to filter breakout direction (long only in uptrend, short only in downtrend).
# Volume spike ensures breakout has conviction. Target 15-30 trades/year to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).
# Avoids false breakouts in ranging markets via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Using yesterday's data to avoid look-ahead
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla calculations
    rang = prev_high - prev_low
    r3 = prev_close + (rang * 1.1000 / 4)
    s3 = prev_close - (rang * 1.1000 / 4)
    r4 = prev_close + (rang * 1.1000 / 2)
    s4 = prev_close - (rang * 1.1000 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R3, above EMA34 (uptrend), volume spike
        if (close[i] > r3_aligned[i] and 
            close[i] > ema34_aligned[i] and 
            volume_spike[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S3, below EMA34 (downtrend), volume spike
        elif (close[i] < s3_aligned[i] and 
              close[i] < ema34_aligned[i] and 
              volume_spike[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to opposite S3/R3 level (mean reversion)
        elif position == 1 and close[i] < s3_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > r3_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HT"
timeframe = "6h"
leverage = 1.0