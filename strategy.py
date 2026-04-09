#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Long when price touches Camarilla H3 level with volume confirmation and chop < 61.8 (trending)
# Short when price touches Camarilla L3 level with volume confirmation and chop < 61.8 (trending)
# Uses discrete position sizing 0.25 to target ~20-40 trades/year and minimize fee drag
# Works in bull/bear markets: pivot levels act as support/resistance, volume confirms breakouts,
# chop filter avoids ranging markets where pivot touches fail

name = "4h_1d_camarilla_pivot_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + (Range * 1.1 / 2)
    # L3 = Pivot - (Range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_h3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    camarilla_l3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Calculate 14-period choppiness index on 1d
    def true_range(high_arr, low_arr, close_arr):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first period
        return tr
    
    tr_1d = true_range(high_1d, low_1d, close_1d)
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_tr_14 / (atr_14_1d * 14)) / np.log10(14)
    chop_1d = np.where(atr_14_1d > 0, chop_1d, 50.0)  # avoid division by zero
    
    # Align 1d indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current 4h volume > 2.0x average 4h volume (20-period)
    vol_s_4h = pd.Series(volume)
    avg_vol_4h = vol_s_4h.rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * avg_vol_4h
    volume_confirmed = np.where(np.isnan(avg_vol_4h), False, volume_confirmed)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(avg_vol_4h[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        if chop_aligned[i] >= 61.8:
            # In ranging market, exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below L3 or reverses from H3
            if close[i] < camarilla_l3_aligned[i] or (close[i] < camarilla_h3_aligned[i] * 0.998):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above H3 or reverses from L3
            if close[i] > camarilla_h3_aligned[i] or (close[i] > camarilla_l3_aligned[i] * 1.002):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long when price touches H3 with volume confirmation
            if (abs(high[i] - camarilla_h3_aligned[i]) < (camarilla_h3_aligned[i] * 0.002) or
                abs(low[i] - camarilla_h3_aligned[i]) < (camarilla_h3_aligned[i] * 0.002)):
                if volume_confirmed[i]:
                    position = 1
                    signals[i] = 0.25
            # Enter short when price touches L3 with volume confirmation
            elif (abs(high[i] - camarilla_l3_aligned[i]) < (camarilla_l3_aligned[i] * 0.002) or
                  abs(low[i] - camarilla_l3_aligned[i]) < (camarilla_l3_aligned[i] * 0.002)):
                if volume_confirmed[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals