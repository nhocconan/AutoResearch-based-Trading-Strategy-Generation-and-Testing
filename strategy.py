#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1wTrendFilter_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts from weekly pivot levels with 1-week EMA50 trend filter and volume spike confirmation on 12h timeframe.
Only long when price breaks above R1 in bull regime (price > 1w EMA50), short when breaks below S1 in bear regime (price < 1w EMA50).
Volume spike > 2.0 * ATR12h confirms momentum. Discrete sizing 0.25 to minimize fee drag.
Target: 12-37 trades/year (50-150 total over 4 years) to avoid overtrading while capturing strong directional moves.
Uses weekly Camarilla levels for structural breaks, 1-week trend filter for regime, and volume spike for confirmation.
Works in bull via breakouts, works in bear via short breakdowns, avoids ranging markets via regime filter.
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
    
    # Get 1w data for weekly Camarilla levels and trend regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend regime
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for volume spike filter (using 12h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly Camarilla levels from previous week's OHLC
    o_1w = df_1w['open'].values
    h_1w = df_1w['high'].values
    l_1w = df_1w['low'].values
    c_1w = df_1w['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1_1w = c_1w + (h_1w - l_1w) * 1.1 / 12
    camarilla_s1_1w = c_1w - (h_1w - l_1w) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1w EMA50 (50) and ATR (14)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0 * ATR (stricter to reduce trades)
        volume_spike = volume[i] > 2.0 * atr[i]
        
        # Determine 1w trend regime
        # Bull regime: price > EMA50
        # Bear regime: price < EMA50
        if close[i] > ema_50_1w_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_50_1w_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades (unlikely but handle)
        
        if position == 0:
            # Long setup: price breaks above R1 AND volume spike AND bull regime
            long_setup = (close[i] > camarilla_r1_aligned[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below S1 AND volume spike AND bear regime
            short_setup = (close[i] < camarilla_s1_aligned[i]) and volume_spike and (regime == 'bear')
            
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
            # Exit: price breaks below S1 OR regime turns bearish
            if (close[i] < camarilla_s1_aligned[i]) or (regime == 'bear'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR regime turns bullish
            if (close[i] > camarilla_r1_aligned[i]) or (regime == 'bull'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1wTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0