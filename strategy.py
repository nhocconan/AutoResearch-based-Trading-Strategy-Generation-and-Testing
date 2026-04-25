#!/usr/bin/env python3
"""
12h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeSpike_v4
Hypothesis: Trade 12h Camarilla H3/L3 breakouts with 1d EMA50 trend filter and volume confirmation.
Only trade breakouts in direction of 1d trend: long when price > EMA50, short when price < EMA50.
Use volume > 2.0 * ATR20 for confirmation to avoid false breakouts.
Target: 12-30 trades/year (50-150 total over 4 years) to minimize fee drag.
Discrete sizing: 0.25.
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
    
    # Get 1d data for trend regime and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend regime
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for volume confirmation (using 12h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50) and ATR (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels using previous 1d bar
        # Camarilla levels are based on previous day's range
        idx_1d = i // 24  # 24 * 12h bars = 1d bar
        if idx_1d < 1:
            signals[i] = 0.0
            continue
            
        # Get previous 1d bar (completed bar)
        prev_high = df_1d['high'].iloc[idx_1d - 1]
        prev_low = df_1d['low'].iloc[idx_1d - 1]
        prev_close = df_1d['close'].iloc[idx_1d - 1]
        
        range_ = prev_high - prev_low
        
        # Camarilla H3 and L3 levels
        h3 = prev_close + range_ * 1.1 / 4
        l3 = prev_close - range_ * 1.1 / 4
        
        # Volume spike: current volume > 2.0 * ATR
        volume_spike = volume[i] > 2.0 * atr[i]
        
        # Determine 1d trend regime
        # Bull regime: price > EMA50
        # Bear regime: price < EMA50
        if close[i] > ema_50_1d_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_50_1d_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades
        
        if position == 0:
            # Long setup: price breaks above H3 AND volume spike AND bull regime
            long_setup = (close[i] > h3) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below L3 AND volume spike AND bear regime
            short_setup = (close[i] < l3) and volume_spike and (regime == 'bear')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below L3 OR regime turns bearish
            if (close[i] < l3) or (regime == 'bear'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above H3 OR regime turns bullish
            if (close[i] > h3) or (regime == 'bull'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_H3L3_Breakout_1dTrendFilter_VolumeSpike_v4"
timeframe = "12h"
leverage = 1.0