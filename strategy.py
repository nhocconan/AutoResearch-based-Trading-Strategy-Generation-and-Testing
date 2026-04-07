#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + Volume + 1d EMA Trend Filter
# Hypothesis: Fade at Camarilla R3/S3 levels in direction of daily EMA(20) trend
# with volume confirmation. Works in bull/bear by trading with daily trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "4h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    close_daily = df_daily['close'].values
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    
    # Previous day's Camarilla levels
    prev_close = np.roll(close_daily, 1)
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla formula
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h
    R3_4h = align_htf_to_ltf(prices, df_daily, R3)
    S3_4h = align_htf_to_ltf(prices, df_daily, S3)
    R4_4h = align_htf_to_ltf(prices, df_daily, R4)
    S4_4h = align_htf_to_ltf(prices, df_daily, S4)
    
    # Daily EMA(20) for trend filter
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(R4_4h[i]) or 
            np.isnan(S4_4h[i]) or np.isnan(ema_20_4h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 or trend changes
            if low[i] <= S3_4h[i] or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches R3 or trend changes
            if high[i] >= R3_4h[i] or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at R3/S3 in direction of daily EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_4h[i]:  # Uptrend
                    if low[i] <= S3_4h[i] and close[i] > S3_4h[i]:  # Bounce off S3
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if high[i] >= R3_4h[i] and close[i] < R3_4h[i]:  # Rejection at R3
                        position = -1
                        signals[i] = -0.25
    
    return signals