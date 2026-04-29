#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses daily Williams Fractals for key swing high/low levels (more robust than Camarilla)
# 1d EMA50 for trend filter to avoid counter-trend trades in ranging/bear markets
# Volume spike (2.0x 20-period average) confirms breakout validity
# Designed for 6h timeframe with tight entries (target: 50-150 total trades over 4 years)
# Works in bull markets via trend-following breaks and in bear markets via trend avoidance

name = "6h_Williams_Fractal_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Fractals on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bearish_fractal, bullish_fractal = compute_williams_fractals(high_1d, low_1d)
    # Williams fractals need 2 extra bars for confirmation (center bar + 2 right bars)
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 2 previous bars for fractal calculation
        if i < 2:
            signals[i] = 0.0
            continue
            
        # Get current 1d aligned values
        curr_close = close[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_bearish = bearish_fractal_aligned[i]
        curr_bullish = bullish_fractal_aligned[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price below bullish fractal OR price below 1d EMA50
            if curr_close < curr_bullish or curr_close < curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above bearish fractal OR price above 1d EMA50
            if curr_close > curr_bearish or curr_close > curr_ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above bullish fractal AND price > 1d EMA50 AND volume spike
            if curr_close > curr_bullish and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below bearish fractal AND price < 1d EMA50 AND volume spike
            elif curr_close < curr_bearish and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals

def compute_williams_fractals(high, low):
    """Compute Williams Fractals - returns (bearish, bullish) arrays"""
    n = len(high)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    
    # Need at least 5 points: 2 left, center, 2 right
    for i in range(2, n-2):
        # Bearish fractal: high[i] is highest among i-2, i-1, i, i+1, i+2
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish[i] = high[i]
        
        # Bullish fractal: low[i] is lowest among i-2, i-1, i, i+1, i+2
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish[i] = low[i]
    
    return bearish, bullish