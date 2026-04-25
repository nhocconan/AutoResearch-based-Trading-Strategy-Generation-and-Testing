#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts with 12h EMA50 trend filter and volume spike confirmation on 4h timeframe.
Only long when price breaks above R1 in bull regime (price > 12h EMA50), short when breaks below S1 in bear regime (price < 12h EMA50).
Volume spike > 1.5 * ATR4h confirms momentum. Discrete sizing 0.25 to minimize fee drift.
Target: 20-40 trades/year to avoid overtrading while capturing strong directional moves.
Uses 12h HTF for better trend alignment than 1d, reducing whipsaw in ranging markets.
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
    
    # Get 12h data for trend regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend regime
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for volume spike filter (using 4h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 12h OHLC for Camarilla levels
    o_12h = df_12h['open'].values
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for each 12h bar
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    camarilla_r1_12h = c_12h + (h_12h - l_12h) * 1.1 / 12
    camarilla_s1_12h = c_12h - (h_12h - l_12h) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1_12h)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 12h EMA50 (50) and ATR (14)
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * ATR
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # Determine 12h trend regime
        # Bull regime: price > EMA50
        # Bear regime: price < EMA50
        if close[i] > ema_50_12h_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_50_12h_aligned[i]:
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

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0