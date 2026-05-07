#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_Volume
Hypothesis: Trade 1-hour breakouts of 4-hour Camarilla R1/S1 levels only when aligned with 4-hour trend (EMA50) and confirmed by volume spike (>3x average). This uses higher timeframe for signal direction and lower timeframe for precise entry timing, reducing trade frequency to 15-30 per year while maintaining edge in both bull and bear markets.
"""

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (using prior 4h bar's OHLC)
    daily_high = df_4h['high'].values
    daily_low = df_4h['low'].values
    daily_close = df_4h['close'].values
    
    camarilla_range = daily_high - daily_low
    r1 = daily_close + (camarilla_range * 1.1 / 12)
    s1 = daily_close - (camarilla_range * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (with 1-bar delay for completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1, additional_delay_bars=1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1, additional_delay_bars=1)
    
    # Get 4h trend filter (EMA50)
    ema_50_4h = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1h volume for confirmation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, daily_close)
        if np.isnan(close_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_4h_aligned[i] > ema_50_4h_aligned[i]
        trend_down = close_4h_aligned[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with upward trend and volume spike
            if (close[i] > r1_aligned[i] and 
                trend_up and 
                vol_ratio[i] > 3.0):
                signals[i] = 0.20
                position = 1
            # Short breakdown: price breaks below S1 with downward trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  trend_down and 
                  vol_ratio[i] > 3.0):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to 4h close or trend turns down
            if close[i] < close_4h_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to 4h close or trend turns up
            if close[i] > close_4h_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals