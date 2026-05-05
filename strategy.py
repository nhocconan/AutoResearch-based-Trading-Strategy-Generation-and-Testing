#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d volume spike and 1h Supertrend filter
# Long when price breaks above 4h Camarilla R4 level AND 1d volume > 2.0x 20-period average AND 1h Supertrend = bullish
# Short when price breaks below 4h Camarilla S4 level AND 1d volume > 2.0x 20-period average AND 1h Supertrend = bearish
# Exit when price crosses 4h Camarilla H3/L3 levels (mean reversion to equilibrium)
# Uses 4h primary timeframe with 1d HTF for volume confirmation and 1h HTF for trend filter
# Volume confirmation ensures breakouts have conviction
# Supertrend provides robust trend following with ATR-based stop
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_Camarilla_R4S4_Breakout_1dVolume_1hSupertrend"
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
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter
    vol_1d = df_1d['volume'].values
    if len(vol_1d) >= 20:
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        volume_filter_1d = vol_1d > (2.0 * vol_ma_20)
    else:
        volume_filter_1d = np.zeros(len(df_1d), dtype=bool)
    
    # Align 1d volume filter to 4h timeframe
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Get 1h data ONCE before loop for Supertrend trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate 1h Supertrend (ATR=10, multiplier=3.0)
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # True Range
    tr1 = np.abs(high_1h - low_1h)
    tr2 = np.abs(high_1h - np.roll(close_1h, 1))
    tr3 = np.abs(low_1h - np.roll(close_1h, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1h + low_1h) / 2 + 3.0 * atr_1h
    basic_lb = (high_1h + low_1h) / 2 - 3.0 * atr_1h
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(close_1h)
    final_lb = np.zeros_like(close_1h)
    supertrend = np.zeros_like(close_1h, dtype=bool)  # True = bullish, False = bearish
    
    for i in range(1, len(close_1h)):
        if basic_ub[i] < final_ub[i-1] or close_1h[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1h[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    for i in range(len(close_1h)):
        if i == 0:
            supertrend[i] = True
        elif supertrend[i-1]:
            supertrend[i] = close_1h[i] <= final_ub[i]
        else:
            supertrend[i] = close_1h[i] >= final_lb[i]
    
    # Align 1h Supertrend to 4h timeframe
    supertrend_1h_aligned = align_htf_to_ltf(prices, df_1h, supertrend)
    
    # Get 4h data ONCE before loop for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r4 = close_4h + (1.1 * (high_4h - low_4h))  # R4 = Close + 1.1*(High-Low)
    camarilla_s4 = close_4h - (1.1 * (high_4h - low_4h))  # S4 = Close - 1.1*(High-Low)
    camarilla_h3 = close_4h + (1.1/2 * (high_4h - low_4h))  # H3 = Close + 1.1/2*(High-Low)
    camarilla_l3 = close_4h - (1.1/2 * (high_4h - low_4h))  # L3 = Close - 1.1/2*(High-Low)
    
    # Align Camarilla levels to 4h timeframe (same df_4h)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i]) or 
            np.isnan(supertrend_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND volume spike AND bullish Supertrend
            if (close[i] > camarilla_r4_aligned[i] and 
                volume_filter_1d_aligned[i] and 
                supertrend_1h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND volume spike AND bearish Supertrend
            elif (close[i] < camarilla_s4_aligned[i] and 
                  volume_filter_1d_aligned[i] and 
                  not supertrend_1h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Camarilla H3 (mean reversion to equilibrium)
            if close[i] < camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Camarilla L3 (mean reversion to equilibrium)
            if close[i] > camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals