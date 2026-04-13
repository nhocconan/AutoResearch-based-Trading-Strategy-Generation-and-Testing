#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout + 1d trend filter + volume confirmation
    # Long when: price breaks above 12h Camarilla H3 level AND price > 1d EMA50 AND volume > 1.8x 20-bar avg
    # Short when: price breaks below 12h Camarilla L3 level AND price < 1d EMA50 AND volume > 1.8x 20-bar avg
    # Exit when: price crosses 12h Camarilla pivot point (mid-level)
    # Uses discrete sizing (0.25) targeting 75-150 total trades over 4 years (19-37/year).
    # 12h timeframe reduces trade frequency vs lower TFs to minimize fee drag.
    # Camarilla pivots from 12h provide intraday structure with statistical significance.
    # 1d EMA50 ensures we only trade with the higher timeframe trend.
    # Volume confirmation filters weak breakouts.
    # Works in bull (breakouts with trend) and bear (only trend-aligned breaks taken).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla calculations (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels (based on previous bar's OHLC)
    # H4 = close + 1.5*(high-low), H3 = close + 1.0*(high-low), H2 = close + 0.5*(high-low)
    # H1 = close + 0.25*(high-low), Pivot = (high+low+close)/3
    # L1 = close - 0.25*(high-low), L2 = close - 0.5*(high-low)
    # L3 = close - 1.0*(high-low), L4 = close - 1.5*(high-low)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h[0] = np.nan
    prev_low_12h[0] = np.nan
    prev_close_12h[0] = np.nan
    
    camarilla_h3_12h = prev_close_12h + 1.0 * (prev_high_12h - prev_low_12h)
    camarilla_l3_12h = prev_close_12h - 1.0 * (prev_high_12h - prev_low_12h)
    camarilla_pivot_12h = (prev_high_12h + prev_low_12h + prev_close_12h) / 3.0
    
    # Align 12h Camarilla levels to 15m timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_12h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_12h)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot_12h)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.8x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.8 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > camarilla_h3_aligned[i]  # break above Camarilla H3
        breakout_down = close[i] < camarilla_l3_aligned[i]  # break below Camarilla L3
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (close[i] > ema_50_1d_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < ema_50_1d_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
        exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_pivot_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0