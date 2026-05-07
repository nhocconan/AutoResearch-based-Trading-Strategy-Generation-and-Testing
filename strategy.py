#!/usr/bin/env python3
# 1d_1w_OBV_Trend_Momentum
# Uses 1-week OBV trend (long-term momentum) + 1d price action + volume confirmation.
# Long when price is above 1d EMA20, OBV rising, and volume above average.
# Short when price below 1d EMA20, OBV falling, and volume above average.
# Exit when price crosses back through 1d EMA20.
# Designed for 1d timeframe to capture multi-day momentum with volume confirmation.

name = "1d_1w_OBV_Trend_Momentum"
timeframe = "1d"
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
    
    # Get weekly data for OBV trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly OBV
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # OBV calculation: cumulative volume with sign based on price change
    price_change = np.diff(close_1w, prepend=close_1w[0])
    obv_direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
    obv = np.cumsum(obv_direction * volume_1w)
    
    # Weekly EMA10 of OBV for trend smoothing
    obv_ema10 = pd.Series(obv).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align OBV EMA10 to daily timeframe
    obv_ema10_aligned = align_htf_to_ltf(prices, df_1w, obv_ema10)
    
    # Daily EMA20 for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily volume filter (20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_above_avg = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # Track holding period
    
    for i in range(30, n):
        # Skip if any critical value is NaN
        if (np.isnan(obv_ema10_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(volume_above_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price above EMA20, OBV trending up, volume confirmation
            if close[i] > ema_20[i] and obv_ema10_aligned[i] > obv_ema10_aligned[i-1] and volume_above_avg[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price below EMA20, OBV trending down, volume confirmation
            elif close[i] < ema_20[i] and obv_ema10_aligned[i] < obv_ema10_aligned[i-1] and volume_above_avg[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price crosses back below EMA20
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry >= 2 and close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses back above EMA20
            # Minimum holding period of 2 days to reduce churn
            if bars_since_entry >= 2 and close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals