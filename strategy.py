# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla pivot levels on 12h combined with 1d EMA trend filter and volume spikes.
# Camarilla levels (R3/S3) act as strong support/resistance with high breakout probability.
# In trending markets (price > 1d EMA34), breakouts above R3 or below S3 have strong follow-through.
# Volume spike confirms institutional participation in breakouts.
# Designed for ~20-30 trades/year per symbol with strong edge in both bull and bear markets.

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
    
    # Calculate Camarilla levels for 12h timeframe
    # Using previous period's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    range_hl = prev_high - prev_low
    r3 = prev_close + (range_hl * 1.1 / 2)
    s3 = prev_close - (range_hl * 1.1 / 2)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2.0x 20-period average (strict for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # In uptrend (price > 1d EMA34), look for long breakouts above R3
        if close[i] > ema34_1d_aligned[i]:
            if close[i] > r3[i] and volume_filter[i]:
                # Long breakout above R3 with volume
                signals[i] = 0.30
                position = 1
            elif close[i] < s3[i] and position == 1:
                # Exit long if price breaks below S3
                signals[i] = 0.0
                position = 0
        # In downtrend (price < 1d EMA34), look for short breakdowns below S3
        elif close[i] < ema34_1d_aligned[i]:
            if close[i] < s3[i] and volume_filter[i]:
                # Short breakdown below S3 with volume
                signals[i] = -0.30
                position = -1
            elif close[i] > r3[i] and position == -1:
                # Exit short if price breaks above R3
                signals[i] = 0.0
                position = 0
        else:
            # Around EMA, hold or flatten
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0