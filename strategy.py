#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter.
    # Long when price breaks above Camarilla H3 (resistance) with volume > 2x average and chop > 61.8 (trending).
    # Short when price breaks below Camarilla L3 (support) with volume > 2x average and chop > 61.8 (trending).
    # Exit when price returns to Camarilla Pivot point or opposite extreme (H4/L4).
    # Uses proven Camarilla structure with volume confirmation and regime filter to minimize false breakouts.
    # Target: 75-200 total trades over 4 years (19-50/year) to balance edge and fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for choppiness indicator (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate choppiness index: CHOP = 100 * LOG10(SUM(ATR14) / (MAX(HIGH,n) - MIN(LOW,n))) / LOG10(n)
    sum_atr14 = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (max_high14 - min_low14 + 1e-10)) / np.log10(14)
    
    # Get 1d data for Camarilla pivot levels (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Pivot + Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # L3 = Pivot - Range * 1.1/4
    # L4 = Pivot - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h4_1d = pivot_1d + range_1d * 1.1 / 2.0
    h3_1d = pivot_1d + range_1d * 1.1 / 4.0
    l3_1d = pivot_1d - range_1d * 1.1 / 4.0
    l4_1d = pivot_1d - range_1d * 1.1 / 2.0
    
    # Align HTF indicators to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Calculate volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop > 61.8 indicates trending market (good for breakouts)
        trending_regime = chop[i] > 61.8
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = (close[i] > h3_1d_aligned[i]) and trending_regime and volume_confirm
        short_breakout = (close[i] < l3_1d_aligned[i]) and trending_regime and volume_confirm
        
        # Exit conditions: return to pivot or opposite extreme
        long_exit = (position == 1) and (close[i] <= pivot_1d_aligned[i] or close[i] >= h4_1d_aligned[i])
        short_exit = (position == -1) and (close[i] >= pivot_1d_aligned[i] or close[i] <= l4_1d_aligned[i])
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif long_exit:
            position = 0
            signals[i] = 0.0
        elif short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_chop_regime_v1"
timeframe = "4h"
leverage = 1.0