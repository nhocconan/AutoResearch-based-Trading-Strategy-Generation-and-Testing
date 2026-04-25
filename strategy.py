#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrendFilter_VolumeSpike_v1
Hypothesis: Trade Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume spike confirmation on 6h timeframe.
R3/S3 levels represent stronger support/resistance than R1/S1, reducing false breakouts.
Only long when price breaks above R3 in bull regime (price > 1d EMA34), short when breaks below S3 in bear regime (price < 1d EMA34).
Volume spike > 1.5 * ATR6h confirms momentum. Discrete sizing 0.25 to minimize fee churn.
Target: 12-25 trades/year to avoid overtrading while capturing strong directional moves in both bull and bear markets.
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend regime
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for volume spike filter (using 6h data)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # first TR undefined
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get 1d OHLC for Camarilla levels
    o_1d = df_1d['open'].values
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Calculate Camarilla R3/S3 levels for each 1d bar
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    camarilla_r3_1d = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_s3_1d = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and ATR (14)
    start_idx = max(34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * ATR
        volume_spike = volume[i] > 1.5 * atr[i]
        
        # Determine 1d trend regime
        # Bull regime: price > EMA34
        # Bear regime: price < EMA34
        if close[i] > ema_34_1d_aligned[i]:
            regime = 'bull'  # only allow longs
        elif close[i] < ema_34_1d_aligned[i]:
            regime = 'bear'  # only allow shorts
        else:
            regime = 'range'  # no trades (unlikely but handle)
        
        if position == 0:
            # Long setup: price breaks above R3 AND volume spike AND bull regime
            long_setup = (close[i] > camarilla_r3_aligned[i]) and volume_spike and (regime == 'bull')
            
            # Short setup: price breaks below S3 AND volume spike AND bear regime
            short_setup = (close[i] < camarilla_s3_aligned[i]) and volume_spike and (regime == 'bear')
            
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
            # Exit: price breaks below S3 OR regime turns bearish
            if (close[i] < camarilla_s3_aligned[i]) or (regime == 'bear'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R3 OR regime turns bullish
            if (close[i] > camarilla_r3_aligned[i]) or (regime == 'bull'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0