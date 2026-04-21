#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_VolumeATRRegime_v1
Hypothesis: Breakout of Camarilla R1/S1 on 4h with 12h trend alignment (EMA34), volume confirmation, and chop regime filter. Works in bull/bear via 12h EMA trend and Camarilla levels for precise entries.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    r2 = prev_close + rang * 2.0 / 12
    s2 = prev_close - rang * 2.0 / 12
    r3 = prev_close + rang * 3.0 / 12
    s3 = prev_close - rang * 3.0 / 12
    r4 = prev_close + rang * 6.0 / 12
    s4 = prev_close - rang * 6.0 / 12
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 12h data for trend (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Choppiness regime filter on 12h (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    if len(df_12h) >= 14:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        atr_14 = []
        tr = np.maximum(high_12h[1:] - low_12h[1:], np.maximum(np.abs(high_12h[1:] - close_12h[:-1]), np.abs(low_12h[1:] - close_12h[:-1])))
        tr = np.concatenate([[np.nan], tr])
        for i in range(len(close_12h)):
            if i < 14:
                atr_14.append(np.nan)
            else:
                atr_14.append(np.nanmean(tr[i-13:i+1]))
        atr_14 = np.array(atr_14)
        sum_tr = np.nansum(atr_14)
        if sum_tr > 0 and not np.isnan(sum_tr):
            chop = 100 * np.log10(sum_tr / (np.max(high_12h) - np.min(low_12h))) / np.log10(14)
            chop_value = chop
        else:
            chop_value = 50.0
        # Simplified chop calculation per bar (approximation for regime)
        chop_series = np.full(len(close_12h), 50.0)
        for i in range(14, len(close_12h)):
            period_high = np.max(high_12h[i-13:i+1])
            period_low = np.min(low_12h[i-13:i+1])
            period_sum_tr = np.nansum(tr[i-13:i+1])
            if period_sum_tr > 0 and (period_high - period_low) > 0:
                chop_series[i] = 100 * np.log10(period_sum_tr / (period_high - period_low)) / np.log10(14)
        chop_aligned = align_htf_to_ltf(prices, df_12h, chop_series)
        chop_ok = chop_aligned < 61.8  # Avoid strong ranging markets
    else:
        chop_ok = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if chop_ok and i < len(chop_aligned) and not chop_ok:
            chop_ok_local = False
        else:
            chop_ok_local = chop_ok if i < len(chop_aligned) else True
        
        if position == 0:
            # Long conditions: break above R1 with 12h uptrend and volume
            if (price > r1_aligned[i] and 
                ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1] and  # 12h EMA rising
                volume_ok and chop_ok_local):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 with 12h downtrend and volume
            elif (price < s1_aligned[i] and 
                  ema_34_12h_aligned[i] < ema_34_12h_aligned[i-1] and  # 12h EMA falling
                  volume_ok and chop_ok_local):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below S1 or reach R2
            if price < s1_aligned[i] or price > r2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1 or reach S2
            if price > r1_aligned[i] or price < s2_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeATRRegime_v1"
timeframe = "4h"
leverage = 1.0