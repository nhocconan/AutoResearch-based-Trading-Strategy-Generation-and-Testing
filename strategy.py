#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter_v1
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakouts with 1-week EMA34 trend filter and choppiness regime filter capture institutional momentum moves while avoiding choppy markets. R1/S1 levels represent strong intraday support/resistance where breaks indicate smart money participation. 1-week EMA34 ensures alignment with long-term trend. Choppiness filter (CHOP > 61.8) avoids range-bound markets. Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Load 1w data ONCE before loop for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First bar: use first available values (no look-ahead)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla pivot calculations
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 12
    camarilla_s1 = prev_close - range_1d * 1.1 / 12
    camarilla_r2 = prev_close + range_1d * 1.1 / 6
    camarilla_s2 = prev_close - range_1d * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Choppiness regime filter on 1d timeframe
    # CHOP(14) = 100 * log10(sum(ATR(1),14) / (log10(HH(14)-LL(14)) * sqrt(14)))
    # Simplified: CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = trending
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    atr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 * 14 / (hh_14 - ll_14)) / np.log10(np.sqrt(14))
    chop_raw = np.nan_to_num(chop_raw, nan=50.0)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Volume spike detection on 12h (volume > 1.8x 30-period EMA)
    volume_ema = pd.Series(volume).ewm(span=30, adjust=False, min_periods=30).mean().values
    volume_spike = volume > (volume_ema * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need sufficient data for all indicators)
    start_idx = max(100, 34, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(chop_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend filter (EMA34)
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        # Regime filter: only trade when NOT choppy (CHOP < 61.8 = trending)
        not_choppy = chop_aligned[i] < 61.8
        
        # Long logic: price breaks above R1 with volume spike + uptrend + not choppy
        if close[i] > r1_aligned[i] and volume_spike[i] and uptrend and not_choppy:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: price breaks below S1 with volume spike + downtrend + not choppy
        elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend and not_choppy:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: price reaches opposite S2/R2 level or trend reversal or choppy market
        elif position == 1 and (close[i] < s2_aligned[i] or not uptrend or chop_aligned[i] >= 61.8):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > r2_aligned[i] or not downtrend or chop_aligned[i] >= 61.8):
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

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_RegimeFilter_v1"
timeframe = "12h"
leverage = 1.0