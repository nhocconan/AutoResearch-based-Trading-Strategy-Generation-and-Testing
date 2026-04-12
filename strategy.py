#!/usr/bin/env python3
"""
12h_1d_camarilla_volume_breakout_v2
Hypothesis: 12-hour strategy using daily Camarilla pivot levels with volume confirmation and volatility filter.
Enters long when price breaks above H3 with volume spike; short when breaks below L3 with volume spike.
Uses volatility-adjusted position sizing to reduce risk in choppy markets. Designed for trending markets with clear breakouts.
Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag while capturing strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels using previous day's data (avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Calculate pivot and range
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels - H3 and L3 are the key levels for entry
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    
    # Align Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate ATR for volatility filter and position sizing
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Volatility-adjusted position size (0.20-0.30 range)
        if i >= 20:
            atr_ma = np.mean(atr[max(0, i-20):i+1])
        else:
            atr_ma = atr[i]
        volatility_factor = np.clip(atr[i] / atr_ma, 0.5, 2.0)
        base_size = 0.25
        position_size = base_size * volatility_factor
        position_size = np.clip(position_size, 0.20, 0.30)
        
        # Entry conditions: Camarilla breakout with volume confirmation
        long_breakout = close[i] > h3_12h[i] and volume_filter
        short_breakout = close[i] < l3_12h[i] and volume_filter
        
        # Exit conditions: reverse breakout
        long_exit = close[i] < l3_12h[i]
        short_exit = close[i] > h3_12h[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_volume_breakout_v2"
timeframe = "12h"
leverage = 1.0