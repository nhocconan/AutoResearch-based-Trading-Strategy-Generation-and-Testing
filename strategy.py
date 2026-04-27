#!/usr/bin/env python3
"""
12h_WilliamsFractal_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Williams Fractal breakouts with daily EMA50 trend filter and volume spikes capture momentum in trending markets. 
In bull markets, upward fractal breaks signal continuation; in bear markets, downward fractal breaks signal reversals.
Volume confirms institutional participation, EMA50 filters counter-trend noise. 
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

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
    
    # Get 1d data for Williams Fractals and EMA50
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Fractals (bearish = sell signal, bullish = buy signal)
    # Bearish fractal: high[n-2] < high[n-1] > high[n] and high[n-1] > high[n-3] and high[n-1] > high[n+1]
    # Bullish fractal: low[n-2] > low[n-1] < low[n] and low[n-1] < low[n-3] and low[n-1] < low[n+1]
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_1d), dtype=bool)
    bullish_fractal = np.zeros(len(low_1d), dtype=bool)
    
    # Need at least 5 points for fractal calculation (2 left, 2 right)
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i-2] < high_1d[i-1] and 
            high_1d[i] < high_1d[i-1] and 
            high_1d[i-3] < high_1d[i-1] and 
            high_1d[i+1] < high_1d[i-1]):
            bearish_fractal[i-1] = True  # Center at i-1
            
        if (low_1d[i-2] > low_1d[i-1] and 
            low_1d[i] > low_1d[i-1] and 
            low_1d[i-3] > low_1d[i-1] and 
            low_1d[i+1] > low_1d[i-1]):
            bullish_fractal[i-1] = True  # Center at i-1
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d := df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # Align Williams Fractals to 12h timeframe with additional delay for confirmation
    # Williams Fractals need 2 extra bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal.astype(float), additional_delay_bars=2)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))  # Volume is LTF but confirm using 1d avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need fractals (5), EMA50 (50), volume avg (20)
    start_idx = max(5, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        bearish = bearish_fractal_aligned[i] > 0.5
        bullish = bullish_fractal_aligned[i] > 0.5
        ema50 = ema50_1d_aligned[i]
        vol_conf = volume_confirm_aligned[i] > 0.5
        
        if position == 0:
            # Determine trend: price vs EMA50 (1d)
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            if uptrend and vol_conf and bullish:
                # Long bias: bullish fractal break in uptrend with volume
                signals[i] = size
                position = 1
                entry_price = close_val
            elif downtrend and vol_conf and bearish:
                # Short bias: bearish fractal break in downtrend with volume
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit: bearish fractal break or price below EMA50
            if bearish or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: bullish fractal break or price above EMA50
            if bullish or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsFractal_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0