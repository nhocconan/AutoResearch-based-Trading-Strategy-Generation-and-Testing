#!/usr/bin/env python3
"""
4h_12h_camarilla_volume_breakout_v1
Hypothesis: 4-hour strategy using daily Camarilla pivot levels with volume confirmation and 12h trend filter.
Enters long when price breaks above H3 with volume spike and 12h uptrend; short when breaks below L3 with volume spike and 12h downtrend.
Uses volatility-adjusted position sizing to reduce risk in choppy markets. Designed for trending markets with clear breakouts.
Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag while capturing strong moves.
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
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation (avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla calculations using previous day's data
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + 1.1 * range_val / 2
    l3 = pivot - 1.1 * range_val / 2
    h4 = pivot + 1.1 * range_val
    l4 = pivot - 1.1 * range_val
    
    # Align Camarilla levels to 4h timeframe
    h3_4h = align_htf_to_ltf(prices, df_1d, h3)
    l3_4h = align_htf_to_ltf(prices, df_1d, l3)
    h4_4h = align_htf_to_ltf(prices, df_1d, h4)
    l4_4h = align_htf_to_ltf(prices, df_1d, l4)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend direction
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
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
        if (np.isnan(h3_4h[i]) or np.isnan(l3_4h[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average (avoid low-volume breakouts)
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.5
        else:
            volume_filter = False
        
        # Trend filter from 12h EMA50
        uptrend_12h = close[i] > ema50_12h_aligned[i]
        downtrend_12h = close[i] < ema50_12h_aligned[i]
        
        # Volatility-adjusted position size (0.15-0.30 range)
        atr_ma = np.mean(atr[max(0, i-20):i+1]) if i >= 20 else atr[i]
        volatility_factor = np.clip(atr[i] / atr_ma, 0.5, 2.0)
        base_size = 0.25
        position_size = base_size * volatility_factor
        position_size = np.clip(position_size, 0.15, 0.30)
        
        # Entry conditions: Camarilla breakout with volume and trend confirmation
        long_breakout = close[i] > h3_4h[i] and volume_filter and uptrend_12h
        short_breakout = close[i] < l3_4h[i] and volume_filter and downtrend_12h
        
        # Exit conditions: reverse breakout or trend change
        long_exit = close[i] < l3_4h[i] or not uptrend_12h
        short_exit = close[i] > h3_4h[i] or not downtrend_12h
        
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

name = "4h_12h_camarilla_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0