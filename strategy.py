#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Uses Camarilla levels from daily data: fade at R3/S3, breakout continuation at R4/S4
# 1d EMA50 filter ensures trades align with daily trend
# Volume confirmation reduces false breakouts
# Designed for 4h timeframe to target 19-50 trades/year (75-200 over 4 years)
# Works in bull/bear: EMA50 adapts to trend, Camarilla provides structure

name = "4h_1d_camarilla_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = close_1d + range_1d * 1.1 / 2
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r4_4h = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_4h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_4h = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_4h = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Load daily data ONCE before loop for EMA50 trend filter
    # Calculate 1d EMA50 with proper min_periods
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-period average volume for volume confirmation
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
        if (np.isnan(r4_4h[i]) or np.isnan(r3_4h[i]) or 
            np.isnan(s3_4h[i]) or np.isnan(s4_4h[i]) or
            np.isnan(ema_50_4h[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion level) OR trend turns bearish
            if close[i] < s3_4h[i] or close[i] < ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion level) OR trend turns bullish
            if close[i] > r3_4h[i] or close[i] > ema_50_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above R4 AND price > 1d EMA50 (bullish trend)
                if close[i] > r4_4h[i] and close[i] > ema_50_4h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below S4 AND price < 1d EMA50 (bearish trend)
                elif close[i] < s4_4h[i] and close[i] < ema_50_4h[i]:
                    position = -1
                    signals[i] = -0.25
                # Mean reversion longs: price crosses above S3 from below in bullish trend
                elif (close[i] > s3_4h[i] and prices['close'].iloc[i-1] <= s3_4h[i] and 
                      close[i] > ema_50_4h[i]):
                    position = 1
                    signals[i] = 0.20
                # Mean reversion shorts: price crosses below R3 from above in bearish trend
                elif (close[i] < r3_4h[i] and prices['close'].iloc[i-1] >= r3_4h[i] and 
                      close[i] < ema_50_4h[i]):
                    position = -1
                    signals[i] = -0.20
    
    return signals