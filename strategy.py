#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike_RegimeFilter
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour EMA50 trend filter, volume confirmation, and choppiness regime filter.
Targets 20-30 trades/year by requiring: 1) price breaks daily R1/S1 levels, 2) aligned with 12h EMA50 trend,
3) volume > 2.0x 20-period average, 4) choppiness index < 50 (trending market). Uses 4h timeframe to balance trade frequency.
Volume spike and regime filters reduce false breakouts in ranging markets. Designed to work in both bull and bear markets by
following the 12h trend direction, avoiding counter-trend entries that fail in volatile/ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for Camarilla pivots (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla R1 and S1 levels (R1 = C + 1.1*(HL/4), S1 = C - 1.1*(HL/4))
    R1 = prev_close + 1.1 * prev_range * (1.0/4.0)
    S1 = prev_close - 1.1 * prev_range * (1.0/4.0)
    
    # Align 1d levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness Index regime filter (14-period) - loaded ONCE for 4h
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(14)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.concatenate([[0], tr1])  # align length
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr14 * 14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    chop_filter = chop < 50.0  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for 1d previous data (1) + 12h EMA50 (50) + volume MA (20) + chop (14)
    start_idx = max(50, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: price relative to 12h EMA50
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        if position == 0:
            # Look for entry signals with volume confirmation, trend alignment, and regime filter
            # Long breakout: price breaks above R1 with uptrend, volume confirmation, and trending regime
            long_breakout = (curr_close > R1_aligned[i]) and uptrend and volume_confirm[i] and chop_filter[i]
            # Short breakout: price breaks below S1 with downtrend, volume confirmation, and trending regime
            short_breakout = (curr_close < S1_aligned[i]) and downtrend and volume_confirm[i] and chop_filter[i]
            
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
            # Exit if price breaks below S1 (mean reversion) or trend changes to downtrend or chop > 60 (ranging)
            if curr_close < S1_aligned[i] or not uptrend or chop[i] > 60.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if price breaks above R1 (mean reversion) or trend changes to uptrend or chop > 60 (ranging)
            if curr_close > R1_aligned[i] or not downtrend or chop[i] > 60.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0