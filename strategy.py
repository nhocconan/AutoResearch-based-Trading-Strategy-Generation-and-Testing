#!/usr/bin/env python3
"""
1h_HTF4h1d_Trend_Pullback_Entry
Hypothesis: Use 4h trend (EMA50) and 1d structure (previous day high/low) to define bias, then enter on 1h pullbacks to EMA20 in trend direction with volume confirmation. Designed for low trade frequency (15-30/year) by requiring strong 4h trend alignment and 1d structure respect. Works in bull/bear markets by following 4h trend while using 1d levels for institutional reference and 1h for precise entry timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for structure levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for structure levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align 1d levels to 1h timeframe (they change only at daily boundaries)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # 1h EMA20 for dynamic pullback entry
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.20  # Position size (20%)
    
    # Warmup: max of calculations (50 for 4h EMA, 20 for 1h EMA/volume)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(high_1d_aligned[i]) or 
            np.isnan(low_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        ema_50_4h_val = ema_50_4h_aligned[i]
        high_1d_val = high_1d_aligned[i]
        low_1d_val = low_1d_aligned[i]
        ema_20_val = ema_20[i]
        vol_spike = volume_spike[i]
        
        # Determine 4h trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_4h = close_val > ema_50_4h_val
        bearish_4h = close_val < ema_50_4h_val
        
        # Determine position relative to 1d structure
        above_yesterday_high = close_val > high_1d_val
        below_yesterday_low = close_val < low_1d_val
        within_yesterday_range = not above_yesterday_high and not below_yesterday_low
        
        # Entry conditions: pullback to EMA20 in 4h trend direction with volume confirmation
        # Long: 4h bullish + price near/yesterday low + pullback to EMA20 + volume
        long_entry = bullish_4h and below_yesterday_low and (close_val <= ema_20_val * 1.001) and vol_spike
        # Short: 4h bearish + price near/yesterday high + pullback to EMA20 + volume
        short_entry = bearish_4h and above_yesterday_high and (close_val >= ema_20_val * 0.999) and vol_spike
        
        # Exit conditions: opposite 1d structure break (break below yesterday low for long, break above yesterday high for short)
        exit_long = below_yesterday_low
        exit_short = above_yesterday_high
        
        # Minimum holding period: 6 bars (to avoid whipsaw)
        min_hold = 6
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "1h_HTF4h1d_Trend_Pullback_Entry"
timeframe = "1h"
leverage = 1.0