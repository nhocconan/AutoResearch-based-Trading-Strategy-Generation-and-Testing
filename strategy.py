#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla pivot breakout with 4h trend filter and volume confirmation
    # Long: price breaks above H3 (resistance) AND volume > 1.3x 20-period average AND price > 4h EMA50
    # Short: price breaks below L3 (support) AND volume > 1.3x 20-period average AND price < 4h EMA50
    # Exit: price returns to pivot point (mean reversion in 1h timeframe)
    # Using 4h for Camarilla pivots (structure) and EMA50 (trend), 1h only for entry timing
    # Discrete position sizing (0.20) to balance return and drawdown
    # Target: 15-37 trades/year (~60-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for Camarilla pivots and EMA50 (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivots (based on previous 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_4h = (high_4h + low_4h + close_4h) / 3
    # RANGE = H - L
    range_4h = high_4h - low_4h
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_4h = close_4h + range_4h * 1.1 / 4
    l3_4h = close_4h - range_4h * 1.1 / 4
    
    # Align 4h Camarilla levels to 1h (wait for completed 4h bar)
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # 4h EMA50 for trend filter
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume confirmation: >1.3x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_4h_aligned[i]) or np.isnan(l3_4h_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 4h EMA50, only short if price < 4h EMA50
        long_trend_ok = close[i] > ema_4h_aligned[i]
        short_trend_ok = close[i] < ema_4h_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend
        long_entry = (close[i] > h3_4h_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < l3_4h_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to pivot (mean reversion)
        long_exit = close[i] < pivot_4h_aligned[i]
        short_exit = close[i] > pivot_4h_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0