#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike
# Camarilla levels from 1d high/low/close provide reversal zones; 1d EMA34 filters trend; volume spike confirms.
# Works in bull/bear: long on R3 breakout + uptrend + volume, short on S3 breakdown + downtrend + volume.
# Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first value remains
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 6
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price breaks above R3 + uptrend + volume
        if (close[i] > camarilla_r3_aligned[i] and  # Breakout above R3
            close[i] > ema34_1d_aligned[i] and      # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short conditions: price breaks below S3 + downtrend + volume
        elif (close[i] < camarilla_s3_aligned[i] and  # Breakdown below S3
              close[i] < ema34_1d_aligned[i] and      # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0