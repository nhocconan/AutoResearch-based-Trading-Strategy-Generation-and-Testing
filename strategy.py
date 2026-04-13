#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
    # Long: price breaks above H3 (resistance) AND volume > 1.5x 20-period average AND price > 1w EMA20
    # Short: price breaks below L3 (support) AND volume > 1.5x 20-period average AND price < 1w EMA20
    # Exit: price returns to pivot point (mean reversion in 12h timeframe)
    # Using 1w for Camarilla pivots (major structure) and EMA20 (trend), 12h only for entry timing
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla pivots and EMA20 (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivots (based on previous 1w bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    # RANGE = H - L
    range_1w = high_1w - low_1w
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_1w = close_1w + range_1w * 1.1 / 4
    l3_1w = close_1w - range_1w * 1.1 / 4
    
    # Align 1w Camarilla levels to 12h (wait for completed 1w bar)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # 1w EMA20 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1w EMA20, only short if price < 1w EMA20
        long_trend_ok = close[i] > ema_1w_aligned[i]
        short_trend_ok = close[i] < ema_1w_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend
        long_entry = (close[i] > h3_1w_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < l3_1w_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to pivot (mean reversion)
        long_exit = close[i] < pivot_1w_aligned[i]
        short_exit = close[i] > pivot_1w_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
leverage = 1.0