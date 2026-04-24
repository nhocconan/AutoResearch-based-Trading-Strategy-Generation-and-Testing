#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h volume spike and 1d ADX regime filter.
- Primary timeframe: 1h for execution, HTF: 4h for volume MA, 1d for ADX trend.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (mean reversion at Camarilla H3/L3).
- Entry: Long when price breaks above Camarilla H3 AND ADX > 25 AND volume spike.
         Short when price breaks below Camarilla L3 AND ADX > 25 AND volume spike.
         In ranging (ADX < 20): Long when price touches Camarilla L3 AND reverses up.
                                Short when price touches Camarilla H3 AND reverses down.
- Exit: Opposite Camarilla breakout or ADX regime shift to ranging.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h).
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], use .index.hour
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for volume MA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels (H3, L3) on 4h
    # Camarilla: based on previous day's range
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # But for intraday, we use the previous 4h bar's range
    # However, since we need the Camarilla levels for the current 1h bar,
    # we calculate them from the completed 4h bar and align to 1h
    # We'll use the 4h close, high, low to compute the levels
    h3 = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low']) / 4
    l3 = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low']) / 4
    
    # Align Camarilla levels to 1h (they are based on completed 4h bar)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3.values)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(df_4h['volume']).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = df_4h['volume'].values > (1.5 * volume_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough bars for ADX and volume MA
    
    for i in range(start_idx, n):
        # Skip if not in session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike_aligned[i]:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price breaks above H3
                    if curr_close > h3_aligned[i]:
                        signals[i] = 0.20
                        position = 1
                    # Bearish breakout: price breaks below L3
                    elif curr_close < l3_aligned[i]:
                        signals[i] = -0.20
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when price touches L3 and shows reversal (close > low)
                    if curr_low <= l3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.20
                        position = 1
                    # Short when price touches H3 and shows reversal (close < high)
                    elif curr_high >= h3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR ADX drops to ranging
            if curr_close < l3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above H3 OR ADX drops to ranging
            if curr_close > h3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hVolume_1dADX_v1"
timeframe = "1h"
leverage = 1.0