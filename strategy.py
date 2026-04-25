#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_VolumeSpike_1wTrend_v1
Hypothesis: Trade 1d Camarilla pivot (R1/S1) breakouts with 1w trend filter and volume confirmation.
Long when price breaks above R1 in 1w uptrend with volume spike, short when breaks below S1 in 1w downtrend.
Use volume > 2.0 * ATR(20) for confirmation to avoid false breakouts.
Target: 15-25 trades/year to minimize fee drag while capturing sustained moves.
Discrete sizing: 0.25.
Works in both bull (breakouts continue) and bear (breakdowns continue) markets via 1w trend filter.
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
    open_prices = prices['open'].values
    
    # Get 1d data for Camarilla pivots (need previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar using previous day's OHLC
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = np.roll(df_1d['close'].values, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    # First bar has no previous data
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    rng = prev_high - prev_low
    R1 = prev_close + 1.1 * rng / 12
    S1 = prev_close - 1.1 * rng / 12
    
    # Align 1d Camarilla levels to 1d timeframe (no additional delay needed as they're based on prev day)
    R1_1d = R1
    S1_1d = S1
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR for volume spike filter (using 1d data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF data to 1d timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for ATR (20) and EMA (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 2.0 * ATR (adaptive threshold)
        volume_spike = volume[i] > 2.0 * atr[i]
        
        # Determine 1w trend regime
        # Uptrend: price > EMA34
        # Downtrend: price < EMA34
        # Range: near EMA34 (within 1*ATR)
        if close[i] > ema_34_1w_aligned[i] + atr[i]:
            trend = 'up'  # only allow longs
        elif close[i] < ema_34_1w_aligned[i] - atr[i]:
            trend = 'down'  # only allow shorts
        else:
            trend = 'range'  # no trades
        
        if position == 0:
            # Long setup: price breaks above R1 AND volume spike AND uptrend
            long_setup = (close[i] > R1_aligned[i]) and volume_spike and (trend == 'up')
            
            # Short setup: price breaks below S1 AND volume spike AND downtrend
            short_setup = (close[i] < S1_aligned[i]) and volume_spike and (trend == 'down')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: price breaks below S1 OR trend turns down OR max holding period (20 bars = 20 days)
            if (close[i] < S1_aligned[i]) or (trend == 'down') or (bars_since_entry >= 20):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price breaks above R1 OR trend turns up OR max holding period (20 bars = 20 days)
            if (close[i] > R1_aligned[i]) or (trend == 'up') or (bars_since_entry >= 20):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "1d_Camarilla_Pivot_VolumeSpike_1wTrend_v1"
timeframe = "1d"
leverage = 1.0