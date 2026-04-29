#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume confirmation
# Uses weekly EMA34 for strong trend direction to avoid counter-trend trades
# 1d Camarilla H3/L3 levels provide precise breakout zones with built-in profit targets at L4/H4
# Volume > 1.8x average confirms institutional participation
# Discrete position sizing (0.25) with Camarilla L4/H4 exit for swing capture
# Designed for ~15-35 trades/year on 12h timeframe to minimize fee drag
# Works in bull/bear via 1w trend filter - only trades in direction of weekly EMA34
# Target: BTC/ETH focus with proven Camarilla structure + weekly trend edge

name = "12h_Camarilla_H3L3_Breakout_1wEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range
    range_1d = high_1d - low_1d
    
    # Camarilla levels (focus on H3/L3 for entry, H4/L4 for exit)
    h3_1d = close_1d + range_1d * 1.1 / 6.0
    l3_1d = close_1d - range_1d * 1.1 / 6.0
    h4_1d = close_1d + range_1d * 1.1 / 2.0
    l4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align 1d Camarilla levels to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 30-period average volume for confirmation
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Volume MA and 1w EMA34 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_h3 = h3_1d_aligned[i]
        curr_l3 = l3_1d_aligned[i]
        curr_h4 = h4_1d_aligned[i]
        curr_l4 = l4_1d_aligned[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_vol_ma = vol_ma_30[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below Camarilla L4 (swing completion)
            if curr_close < curr_l4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above Camarilla H4 (swing completion)
            if curr_close > curr_h4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 1.8x 30-period average
            vol_confirm = curr_volume > 1.8 * curr_vol_ma
            
            # Long when price breaks above H3 with 1w EMA34 uptrend and volume confirmation
            if curr_high > curr_h3 and curr_close > curr_ema34_1w and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below L3 with 1w EMA34 downtrend and volume confirmation
            elif curr_low < curr_l3 and curr_close < curr_ema34_1w and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals