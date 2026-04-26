#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Trend_Reversal_v1
Hypothesis: Trade reversions from Camarilla H3/L3 levels with 4h EMA50 trend filter and volume confirmation. In ranging markets (ADX<25), fade extreme levels; in trending markets (ADX>25), trade pullbacks to EMA50. Designed for low trade frequency (<50/year) to minimize fee drag while capturing mean reversion in ranges and trend continuation in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3, H4, L4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    H3 = PP + range_1d * 1.1 / 4.0
    L3 = PP - range_1d * 1.1 / 4.0
    H4 = PP + range_1d * 1.1 / 2.0
    L4 = PP - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 4h EMA50 for trend filter
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ADX(14) for regime detection
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(50, 20, 14, 14)  # EMA50, volume avg, ATR, ADX
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema50[i]) or np.isnan(adx[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry signals
            # Regime detection
            is_trending = adx[i] > 25
            is_ranging = adx[i] <= 25
            
            if is_ranging:
                # In ranging markets: fade extreme levels (H4/L4)
                long_entry = (close_val < L4_aligned[i]) and volume_confirm[i]
                short_entry = (close_val > H4_aligned[i]) and volume_confirm[i]
            else:
                # In trending markets: trade pullbacks to EMA50
                # Uptrend: price > EMA50, long on pullback to EMA50
                # Downtrend: price < EMA50, short on pullback to EMA50
                if close_val > ema50[i]:  # Uptrend
                    long_entry = (close_val <= ema50[i] * 1.005) and volume_confirm[i]  # Near EMA50
                    short_entry = False
                else:  # Downtrend
                    long_entry = False
                    short_entry = (close_val >= ema50[i] * 0.995) and volume_confirm[i]  # Near EMA50
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit conditions
            # Take profit at opposite Camarilla level or stoploss
            is_trending = adx[i] > 25
            if is_trending:
                # In trend: trail with EMA50 or exit at H3
                exit_condition = (close_val < ema50[i] * 0.995) or (close_val >= H3_aligned[i])
            else:
                # In range: exit at H3 or mean reversion to PP
                exit_condition = (close_val >= H3_aligned[i]) or (close_val >= PP[i] * 0.995)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit conditions
            is_trending = adx[i] > 25
            if is_trending:
                # In trend: trail with EMA50 or exit at L3
                exit_condition = (close_val > ema50[i] * 1.005) or (close_val <= L3_aligned[i])
            else:
                # In range: exit at L3 or mean reversion to PP
                exit_condition = (close_val <= L3_aligned[i]) or (close_val <= PP[i] * 1.005)
            if exit_condition:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_Pivot_Trend_Reversal_v1"
timeframe = "4h"
leverage = 1.0