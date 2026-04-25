#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_1wTrendFilter_VolumeSpike
Hypothesis: Trade daily Camarilla pivot breakouts (H3/L3) with weekly trend filter and volume confirmation.
Only trade in direction of weekly trend: long in weekly bull regime, short in weekly bear regime.
Use volume spike (>2.0 * ATR) to confirm breakout strength. Target: 15-25 trades/year to minimize fee drag.
Discrete sizing: 0.25.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Camarilla: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    hl_range = df_1d['high'].values - df_1d['low'].values
    camarilla_h3 = df_1d['close'].values + 1.1 * hl_range / 2.0
    camarilla_l3 = df_1d['close'].values - 1.1 * hl_range / 2.0
    
    # Align 1d Camarilla levels to 1d timeframe (no additional delay needed)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend regime
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for volume spike filter (using 1d data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for weekly EMA34 (34)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Determine weekly trend regime
        # Bull regime: price > weekly EMA34
        # Bear regime: price < weekly EMA34
        # Range regime: near weekly EMA34 (within 1.0*ATR)
        regime_threshold = 1.0 * atr[i]
        
        if close[i] > ema_34_1w_aligned[i] + regime_threshold:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1w_aligned[i] - regime_threshold:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        if position == 0:
            # Long setup: price breaks above H3 AND volume spike AND bull regime
            volume_spike = volume[i] > 2.0 * atr[i]
            long_setup = (close[i] > camarilla_h3_aligned[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below L3 AND volume spike AND bear regime
            short_setup = (close[i] < camarilla_l3_aligned[i]) and volume_spike and (regime == 'bear')
            
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
            # Exit: price closes below L3 OR regime turns bearish OR max holding period (20 bars = ~20 days)
            if (close[i] < camarilla_l3_aligned[i]) or (regime == 'bear') or (bars_since_entry >= 20):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: price closes above H3 OR regime turns bullish OR max holding period (20 bars = ~20 days)
            if (close[i] > camarilla_h3_aligned[i]) or (regime == 'bull') or (bars_since_entry >= 20):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_1wTrendFilter_VolumeSpike"
timeframe = "1d"
leverage = 1.0