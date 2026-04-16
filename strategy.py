#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Supertrend trend filter with Donchian(20) breakout and volume confirmation.
# Long when 1w Supertrend is bullish, price breaks above Donchian(20) high, and volume > 1.5x 20-day average.
# Short when 1w Supertrend is bearish, price breaks below Donchian(20) low, and volume > 1.5x 20-day average.
# Exit when Supertrend flips or price touches opposite Donchian band.
# Uses discrete position size 0.25. Supertrend provides robust trend, Donchian captures breakouts,
# volume filter ensures momentum confirmation. Target: 30-100 total trades over 4 years (7-25/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Supertrend (ATR=10, mult=3.0) ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2.0 + 3.0 * atr_10
    basic_lb = (high_1w + low_1w) / 2.0 - 3.0 * atr_10
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_1w)
    final_lb = np.zeros_like(close_1w)
    supertrend = np.zeros_like(close_1w)
    trend = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    supertrend[0] = final_lb[0]
    trend[0] = 1
    
    for i in range(1, len(close_1w)):
        # Final Upper Band
        if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
        
        # Final Lower Band
        if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
        
        # Supertrend
        if trend[i-1] == 1 and close_1w[i] <= final_ub[i]:
            trend[i] = -1
        elif trend[i-1] == -1 and close_1w[i] >= final_lb[i]:
            trend[i] = 1
        else:
            trend[i] = trend[i-1]
        
        supertrend[i] = final_ub[i] if trend[i] == -1 else final_lb[i]
    
    # Supertrend direction (1=uptrend, -1=downtrend)
    supertrend_dir = trend
    
    # Align 1w Supertrend and direction to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir)
    
    # === 1d Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian Channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume Moving Average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 40  # enough for Donchian(20) and 1w Supertrend
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        st_dir = supertrend_dir_aligned[i]
        st_val = supertrend_aligned[i]
        dh = highest_high[i]
        dl = lowest_low[i]
        vol_ma = vol_ma_20[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Supertrend turns bearish or price touches Donchian low
            if st_dir <= 0 or price <= dl:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Supertrend turns bullish or price touches Donchian high
            if st_dir >= 0 or price >= dh:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: Supertrend direction must be non-zero
            trend_filter = st_dir != 0
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Supertrend bullish, price > Donchian high, volume spike
            if (st_dir > 0) and (price > dh) and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Supertrend bearish, price < Donchian low, volume spike
            elif (st_dir < 0) and (price < dl) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wSupertrend_Donchian20_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0