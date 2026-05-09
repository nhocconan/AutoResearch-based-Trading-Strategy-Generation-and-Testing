# 6h Camarilla R3/S3 Breakout with Volume Spike and 1d EMA Trend Filter
# Strategy targets breakouts of daily Camarilla R3/S3 levels with volume confirmation
# and trend alignment from daily EMA50. Works in both bull and bear markets by
# following momentum when price breaks key daily levels with institutional volume.
# Uses 60-minute bars for execution with 1-day reference for levels and trend.
# Target: 15-35 trades/year per symbol to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's close for Camarilla calculation (R3, S3)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels (R3, S3)
    r3 = prev_close + 1.1 * (prev_high - prev_low) * 3 / 4
    s3 = prev_close - 1.1 * (prev_high - prev_low) * 3 / 4
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema50_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_6h[i]
        s3_val = s3_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above R3 with volume and above trend
            if close[i] > r3_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S3 with volume and below trend
            elif close[i] < s3_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S3 (mean reversion to center)
            if close[i] < s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R3 (mean reversion to center)
            if close[i] > r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals