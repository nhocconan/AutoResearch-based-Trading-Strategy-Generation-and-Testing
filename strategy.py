#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d EMA200 trend filter and volume confirmation
    # Long: price breaks above H3 AND volume > 1.5x 20-period average AND close > 1d EMA200
    # Short: price breaks below L3 AND volume > 1.5x 20-period average AND close < 1d EMA200
    # Exit: price returns to pivot point (mean reversion)
    # Uses 1d for structure (Camarilla pivots + EMA200 trend) and 4h for precise entry timing
    # Discrete position sizing (0.25) to limit drawdown and reduce fee churn
    # Target: 20-50 trades/year (~80-200 over 4 years) to avoid fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA200 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivots (based on previous 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # PIVOT = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    # RANGE = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels:
    # H3 = C + RANGE * 1.1/4
    # L3 = C - RANGE * 1.1/4
    h3_1d = close_1d + range_1d * 1.1 / 4
    l3_1d = close_1d - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to 4h (wait for completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # 1d EMA200 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if close > 1d EMA200, only short if close < 1d EMA200
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Camarilla breakout + volume + trend
        long_entry = (close[i] > h3_1d_aligned[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < l3_1d_aligned[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to pivot (mean reversion)
        long_exit = close[i] < pivot_1d_aligned[i]
        short_exit = close[i] > pivot_1d_aligned[i]
        
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

name = "4h_1d_camarilla_breakout_volume_trend_v2"
timeframe = "4h"
leverage = 1.0