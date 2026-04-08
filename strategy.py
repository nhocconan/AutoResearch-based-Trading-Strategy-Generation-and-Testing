#!/usr/bin/env python3
# 1d_camarilla_pivot_1w_trend_volume_v1
# Hypothesis: Daily Camarilla pivot levels with weekly trend filter and volume confirmation.
# In trending markets, price tends to revert to mean between H3 and L3 levels.
# In ranging markets, price reacts at H4/L4 levels. Weekly trend filter avoids counter-trend trades.
# Volume confirmation reduces false signals. Designed for 1d timeframe to capture multi-day moves.
# Target: 30-100 trades over 4 years (~7-25/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly trend using EMA on weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Calculate average volume for confirmation (20-day)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(weekly_ema_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Calculate daily Camarilla pivot levels
        # Using previous day's OHLC
        if i == 0:
            continue
            
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        # Camarilla levels
        H4 = pivot + (range_val * 1.1 / 2)
        H3 = pivot + (range_val * 1.1 / 4)
        L3 = pivot - (range_val * 1.1 / 4)
        L4 = pivot - (range_val * 1.1 / 2)
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_ok = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below H3 or weekly trend turns bearish
            if close[i] < H3 or close[i] < weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above L3 or weekly trend turns bullish
            if close[i] > L3 or close[i] > weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if not volume_ok:
                signals[i] = 0.0
                continue
                
            # Long entry: price touches L3 or L4 in uptrend
            if (close[i] <= L3 and close[i] > L4) and close[i] > weekly_ema_aligned[i]:
                # Additional confirmation: price is above weekly EMA (uptrend)
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3 or H4 in downtrend
            elif (close[i] >= H3 and close[i] < H4) and close[i] < weekly_ema_aligned[i]:
                # Additional confirmation: price is below weekly EMA (downtrend)
                position = -1
                signals[i] = -0.25
    
    return signals