#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1
Hypothesis: On 4h timeframe, Camarilla R1/S1 breakouts from the prior 1d session, filtered by 1d EMA34 trend and choppiness regime (range-bound when CHOP > 61.8), capture intraday continuation moves. R1/S1 levels represent intraday support/resistance where breaks indicate short-term momentum. Volume spike confirms participation. Choppiness filter avoids whipsaws in strong trends where Camarilla levels lose relevance. Target: 75-150 total trades over 4 years (19-38/year).
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
    
    # Load 1d data ONCE before loop for HTF trend and regime filters
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d ATR14 for choppiness regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr])  # align length
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # Using 14-period lookback
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla R1 and S1 levels from previous 1d bar
    # Camarilla: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    rang = high_1d - low_1d
    r1 = close_1d + (rang * 1.1 / 12)
    s1 = close_1d - (rang * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike detection on 4h (volume > 1.8x 20-period EMA)
    volume_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (volume_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(50, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1d trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Regime filter: only trade when market is ranging (CHOP > 61.8)
        in_range = chop_aligned[i] > 61.8
        
        # Long logic: price breaks above R1 with volume spike + in uptrend + ranging regime
        if close[i] > r1_aligned[i] and volume_spike[i] and uptrend and in_range:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S1 with volume spike + in downtrend + ranging regime
        elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend and in_range:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price returns to opposite Camarilla level or trend/regime changes
        elif position == 1 and (close[i] < s1_aligned[i] or not uptrend or not in_range):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r1_aligned[i] or not downtrend or not in_range):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_RegimeFilter_v1"
timeframe = "4h"
leverage = 1.0