#!/usr/bin/env python3
"""
12h_1w_1d_camarilla_volume_breakout
Hypothesis: 12-hour strategy using weekly and daily price structure for high-probability breakouts.
Enters long when price breaks above weekly H3 with daily volume confirmation and 1-week uptrend;
short when breaks below weekly L3 with volume confirmation and 1-week downtrend.
Uses weekly ATR for volatility filtering and position sizing to reduce risk in choppy periods.
Designed for trending markets with clear structural breaks. Target: 15-30 trades/year (60-120 total over 4 years).
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
    
    # Get weekly data for structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly structure (previous week's data to avoid look-ahead)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    
    # Weekly pivot and range
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    range_1w = prev_high_1w - prev_low_1w
    
    # Weekly Camarilla levels (H3/L3 for breakouts)
    h3_1w = pivot_1w + 1.1 * range_1w / 2
    l3_1w = pivot_1w - 1.1 * range_1w / 2
    
    # Align weekly levels to 12h timeframe
    h3_1w_12h = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_12h = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Weekly ATR for volatility filter and position sizing
    tr1_w = np.abs(high_1w - low_1w)
    tr2_w = np.abs(np.roll(high_1w, 1) - close_1w)
    tr3_w = np.abs(np.roll(low_1w, 1) - close_1w)
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_w_12h = align_htf_to_ltf(prices, df_1w, atr_w)
    
    # Daily volume for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_1d_12h = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Weekly trend filter (EMA50)
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_12h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1w_12h[i]) or np.isnan(l3_1w_12h[i]) or 
            np.isnan(ema50_1w_12h[i]) or np.isnan(atr_w_12h[i]) or 
            np.isnan(vol_ma_1d_12h[i]) or np.isnan(volume_1d_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: daily volume > 1.5x 20-day average
        volume_filter = volume_1d_12h[i] > vol_ma_1d_12h[i] * 1.5
        
        # Trend filter from weekly EMA50
        uptrend_1w = close[i] > ema50_1w_12h[i]
        downtrend_1w = close[i] < ema50_1w_12h[i]
        
        # Volatility-adjusted position size (0.20-0.30 range)
        atr_ma_w = np.mean(atr_w_12h[max(0, i-20):i+1]) if i >= 20 else atr_w_12h[i]
        volatility_factor = np.clip(atr_w_12h[i] / atr_ma_w, 0.5, 2.0)
        base_size = 0.25
        position_size = base_size * volatility_factor
        position_size = np.clip(position_size, 0.20, 0.30)
        
        # Entry conditions: Weekly H3/L3 breakout with volume and trend confirmation
        long_breakout = close[i] > h3_1w_12h[i] and volume_filter and uptrend_1w
        short_breakout = close[i] < l3_1w_12h[i] and volume_filter and downtrend_1w
        
        # Exit conditions: reverse breakout or trend change
        long_exit = close[i] < l3_1w_12h[i] or not uptrend_1w
        short_exit = close[i] > h3_1w_12h[i] or not downtrend_1w
        
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

name = "12h_1w_1d_camarilla_volume_breakout"
timeframe = "12h"
leverage = 1.0