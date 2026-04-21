#!/usr/bin/env python3
"""
4h_Choppiness_Regime_Camarilla_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout filtered by 1d choppiness regime (CHOP>61.8 = range, CHOP<38.2 = trend) and volume spike.
In trending 1d market (CHOP<38.2): breakout continuation (long above R1, short below S1).
In ranging 1d market (CHOP>61.8): mean reversion at Camarilla H3/L3 levels.
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to balance returns and fee drag.
Timeframe: 4h, uses 1d HTF for regime and Camarilla pivots.
Target: 75-200 total trades over 4 years = 19-50/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla and chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r1_1d = df_1d_close + 0.275 * range_1d
    s1_1d = df_1d_close - 0.275 * range_1d
    h3_1d = df_1d_close + 1.1 * range_1d / 2
    l3_1d = df_1d_close - 1.1 * range_1d / 2
    h4_1d = df_1d_high
    l4_1d = df_1d_low
    
    # Calculate 1d choppiness index (CHOP)
    atr_1d_list = []
    for j in range(len(df_1d)):
        if j < 14:
            atr_1d_list.append(np.nan)
        else:
            high_14 = df_1d_high[j-13:j+1]
            low_14 = df_1d_low[j-13:j+1]
            close_14 = df_1d_close[j-13:j+1]
            tr_list = []
            for k in range(1, 14):
                tr = max(
                    high_14[k] - low_14[k],
                    abs(high_14[k] - close_14[k-1]),
                    abs(low_14[k] - close_14[k-1])
                )
                tr_list.append(tr)
            atr_1d = np.mean(tr_list) if tr_list else 0
            atr_1d_list.append(atr_1d)
    atr_1d = np.array(atr_1d_list)
    
    highest_high_14 = pd.Series(df_1d_high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d_low).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop_1d = np.where(
        chop_denom != 0,
        100 * np.log10(atr_1d * np.sqrt(14) / chop_denom) / np.log10(100),
        50.0
    )
    
    # Align 1d indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === Volume spike filter (volume > 1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i])
            or np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i])
            or np.isnan(chop_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        h3 = h3_1d_aligned[i]
        l3 = l3_1d_aligned[i]
        h4 = h4_1d_aligned[i]
        l4 = l4_1d_aligned[i]
        chop = chop_1d_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if chop < 38.2:  # Trending 1d regime - breakout continuation
                long_condition = (price > r1) and vol_spike
                short_condition = (price < s1) and vol_spike
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif chop > 61.8:  # Ranging 1d regime - mean reversion at H3/L3
                long_condition = (price <= l3) and vol_spike
                short_condition = (price >= h3) and vol_spike
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime/exit logic
            elif chop < 38.2:  # Trending - exit on re-entry to R1/S1 zone
                if price < r1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop > 61.8:  # Ranging - exit at opposite H3/L3 or H4/L4
                if price >= h3 or price <= l4:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Transition regime - hold until clear signal
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime/exit logic
            elif chop < 38.2:  # Trending - exit on re-entry to R1/S1 zone
                if price > s1:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop > 61.8:  # Ranging - exit at opposite H3/L3 or H4/L4
                if price <= l3 or price >= h4:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Transition regime - hold until clear signal
                signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Regime_Camarilla_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0