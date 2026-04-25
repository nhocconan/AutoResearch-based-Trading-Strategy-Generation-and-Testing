#!/usr/bin/env python3
"""
6h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Reduce trade frequency by tightening volume confirmation and adding ADX regime filter.
Targets 15-25 trades/year by requiring: 1) price breaks daily H3/L3 levels, 2) aligned with 1d EMA34 trend,
3) volume > 2.5x 20-period average (tighter), 4) ADX(14) > 25 on 6h for trending markets only.
Uses 6h timeframe to minimize fee drag while capturing significant moves in both bull and bear markets.
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla H3 and L3 levels (H3 = C + 1.1*(HL/2), L3 = C - 1.1*(HL/2))
    H3 = prev_close + 1.1 * prev_range * (1.0/2.0)
    L3 = prev_close - 1.1 * prev_range * (1.0/2.0)
    
    # Align 1d levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Volume confirmation: current volume > 2.5 * 20-period average (tighter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.5)
    
    # ADX(14) on 6h for trend filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - low[:-1])), np.abs(low[1:] - high[:-1]))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr = np.concatenate([[np.nan], atr])  # align length
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx = np.concatenate([[np.nan] * 13, adx[13:]])  # align length
    adx_strong = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d EMA34 (34) and previous day data (1)
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and strong ADX
            # Long breakout: price breaks above H3 with uptrend, volume confirmation, and strong trend
            long_breakout = (curr_close > H3_aligned[i]) and uptrend and volume_confirm[i] and adx_strong[i]
            # Short breakout: price breaks below L3 with downtrend, volume confirmation, and strong trend
            short_breakout = (curr_close < L3_aligned[i]) and downtrend and volume_confirm[i] and adx_strong[i]
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if price breaks below L3 (mean reversion) or trend changes or ADX weakens
            if curr_close < L3_aligned[i] or not uptrend or not adx_strong[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above H3 (mean reversion) or trend changes or ADX weakens
            if curr_close > H3_aligned[i] or not downtrend or not adx_strong[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0