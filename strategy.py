#!/usr/bin/env python3
"""
4h_4x4_KC_Breakout_SpikeAndVolatility
Hypothesis: 4x4 Keltner Channel breakouts combined with volume spikes and volatility filters provide robust trend-following signals across bull and bear markets. The 4x4 width reduces whipsaws while capturing strong moves, and volatility filtering ensures trades occur only during meaningful market moves.
"""

name = "4h_4x4_KC_Breakout_SpikeAndVolatility"
timeframe = "4h"
leverage = 1.0

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
    
    # Volume spike: >1.5x 20-period average (balanced frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # ATR for Keltner Channel
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4x4 Keltner Channel (ATR multiplier = 4)
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20 + (4 * atr)
    kc_lower = ema20 - (4 * atr)
    
    # Daily trend filter: 50 EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Daily volatility filter: ATR ratio (current/20-day average) > 0.8
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d[1:] - low_1d[1:]
    tr2_1d = np.abs(high_1d[1:] - close_1d[:-1])
    tr3_1d = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio_1d = atr_1d / atr_ma_1d
    volatility_filter = atr_ratio_1d > 0.8
    volatility_filter_aligned = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volatility_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper KC + daily uptrend + volume spike + volatility filter
            if (close[i] > kc_upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i] and 
                volatility_filter_aligned[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below lower KC + daily downtrend + volume spike + volatility filter
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i] and 
                  volatility_filter_aligned[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower KC or daily trend turns down
            if (close[i] < kc_lower[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price closes above upper KC or daily trend turns up
            if (close[i] > kc_upper[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals