#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout with 4h Trend and Volume Confirmation
# Long when: Price breaks above Camarilla R3 (4h) AND 4h close > 4h EMA50 AND 1h volume > 1.5 * 20-bar avg volume
# Short when: Price breaks below Camarilla S3 (4h) AND 4h close < 4h EMA50 AND 1h volume > 1.5 * 20-bar avg volume
# Exit when price returns to Camarilla Pivot Point (4h) OR opposite Camarilla level touched
# Uses 4h for signal direction and structure, 1h only for precise entry timing
# Volume filter reduces false breakouts, trend filter ensures momentum alignment
# Discrete sizing 0.20 to minimize fee churn, target 60-150 trades over 4 years (15-37/year)
# Works in bull markets via breakout continuation and bear markets via mean reversion off pivots

name = "1h_Camarilla_R3S3_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla pivots and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivots for 4h (based on previous 4h bar)
    # Camarilla levels: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + 1.1 * Range / 2, S3 = Pivot - 1.1 * Range / 2
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3_4h = pivot_4h + (1.1 * range_4h / 2.0)
    s3_4h = pivot_4h - (1.1 * range_4h / 2.0)
    
    # Align Camarilla levels to 1h (wait for completed 4h bar)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume filter: volume > 1.5 * 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(pivot_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with 4h bullish trend and volume confirmation
            if (close[i] > r3_4h_aligned[i] and 
                close_4h[-1] > ema_50_4h[-1] if len(close_4h) > 0 else False and  # Use last known 4h close
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 with 4h bearish trend and volume confirmation
            elif (close[i] < s3_4h_aligned[i] and 
                  close_4h[-1] < ema_50_4h[-1] if len(close_4h) > 0 else False and  # Use last known 4h close
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: return to pivot point OR touch S3 (mean reversion)
            if close[i] < pivot_4h_aligned[i] or close[i] < s3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: return to pivot point OR touch R3 (mean reversion)
            if close[i] > pivot_4h_aligned[i] or close[i] > r3_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals