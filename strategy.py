# 6h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeFilter
# Strategy: Breakout at Camarilla R3/S3 levels with 1d EMA50 trend filter and volume confirmation.
# Exit on mean reversion to the Camarilla Pivot point.
# Timeframe: 6h (primary), HTF: 1d for pivot and trend.
# Rationale: Camarilla levels act as intraday support/resistance; breaks with trend and volume
# indicate strong momentum. Mean reversion to pivot captures reversion after extended moves.
# Designed for 12-37 trades/year to minimize fee drag while capturing strong trends in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous day (to avoid look-ahead)
    camarilla_pivot = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    
    # Calculate daily values first
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    daily_r3 = close_1d + daily_range * 1.1 / 4
    daily_s3 = close_1d - daily_range * 1.1 / 4
    
    # Align to 6h timeframe (previous day's levels for current period)
    camarilla_pivot = align_htf_to_ltf(prices, df_1d, daily_pivot)
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, daily_r3)
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, daily_s3)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above R3, above 1d EMA50, volume spike
        if (close[i] > camarilla_r3[i] and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below S3, below 1d EMA50, volume spike
        elif (close[i] < camarilla_s3[i] and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to Camarilla Pivot (mean reversion)
        elif position == 1 and close[i] < camarilla_pivot[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > camarilla_pivot[i]:
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

name = "6h_Camarilla_R3_S3_Breakout_1dEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0