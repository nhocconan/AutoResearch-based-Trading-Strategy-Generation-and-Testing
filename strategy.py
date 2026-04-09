#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Uses Camarilla levels from 1d data: breakout above R4 = long, below S4 = short
# 1d EMA50 trend filter ensures trades align with higher timeframe trend
# Volume confirmation reduces false breakouts
# Designed for 6h timeframe to target 12-37 trades/year (50-150 over 4 years)
# Works in bull/bear: EMA50 adapts to trend, Camarilla provides robust structure

name = "6h_1d_camarilla_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels (based on previous day's range)
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.25 * (high - low)
    # S3 = close - 1.25 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        high_low_range = high_1d[i-1] - low_1d[i-1]
        camarilla_r4[i] = close_1d[i-1] + 1.5 * high_low_range
        camarilla_s4[i] = close_1d[i-1] - 1.5 * high_low_range
    
    # Align 1d Camarilla levels to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume confirmation (6h volume)
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S4 OR trend turns bearish
            if close[i] < s4_6h[i] or close[i] < ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R4 OR trend turns bullish
            if close[i] > r4_6h[i] or close[i] > ema_50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above R4 AND price > 1d EMA50 (bullish trend)
                if close[i] > r4_6h[i] and close[i] > ema_50_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below S4 AND price < 1d EMA50 (bearish trend)
                elif close[i] < s4_6h[i] and close[i] < ema_50_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals