#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with volume spike and 4h EMA50 trend alignment capture institutional moves.
Trades only during 08-20 UTC to avoid low-liquidity periods. Uses 4h for signal direction (EMA50 trend) and 1h for precise entry timing.
Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe with session filter reducing noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the period"""
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    r1 = close + (range_hl * 1.1 / 12)
    s1 = close - (range_hl * 1.1 / 12)
    
    return r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA50 trend filter and Camarilla levels (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 trend filter
    ema_50_4h = calculate_ema(df_4h['close'].values, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Camarilla levels from 4h data (for 1h breakout signals)
    r1_4h, s1_4h = calculate_camarilla(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values
    )
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA (50) + volume MA (20)
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or \
           np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or \
           np.isnan(s1_4h_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + 4h EMA50 trend alignment
            long_entry = (curr_close > r1_4h_aligned[i]) and volume_spike[i] and (curr_close > ema_50_4h_aligned[i])
            short_entry = (curr_close < s1_4h_aligned[i]) and volume_spike[i] and (curr_close < ema_50_4h_aligned[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price closes below R1 or trend turns bearish
            if curr_close < r1_4h_aligned[i] or curr_close < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit when price closes above S1 or trend turns bullish
            if curr_close > s1_4h_aligned[i] or curr_close > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0