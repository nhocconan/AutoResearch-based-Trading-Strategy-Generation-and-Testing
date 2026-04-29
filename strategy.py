#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 1.5x 20-bar avg
# Exit when price retouches the pivot point (mean reversion to equilibrium)
# Uses discrete position sizing (0.20) to minimize fee churn.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Camarilla pivots identify key intraday support/resistance levels; R3/S3 are strong breakout levels.
# 4h EMA50 filters counter-trend moves, volume confirmation ensures institutional participation.
# Works in bull markets (buying R3 breakouts) and bear markets (selling S3 breakdowns).

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (avoid Asian session noise)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivots for 4h timeframe
    # Pivot = (High + Low + Close) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    # Range = High - Low
    range_4h = high_4h - low_4h
    # R3 = Close + (High - Low) * 1.1/2
    r3_4h = close_4h + range_4h * 1.1 / 2.0
    # S3 = Close - (High - Low) * 1.1/2
    s3_4h = close_4h - range_4h * 1.1 / 2.0
    # Align Camarilla levels to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Volume MA warmup and EMA50 alignment
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pivot_4h_aligned[i]) or 
            np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_4h_aligned[i]
        pivot = pivot_4h_aligned[i]
        r3 = r3_4h_aligned[i]
        s3 = s3_4h_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retouches pivot point (mean reversion)
            if close[i] <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price retouches pivot point (mean reversion)
            if close[i] >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 4h EMA50 AND volume confirmation
            if high[i] > r3 and close[i] > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 AND close < 4h EMA50 AND volume confirmation
            elif low[i] < s3 and close[i] < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
    
    return signals