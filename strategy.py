#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter.
    # Long when price breaks above H3 pivot level AND 1d volume > 1.8x 20-period MA AND chop > 61.8 (range regime).
    # Short when price breaks below L3 pivot level AND 1d volume > 1.8x 20-period MA AND chop > 61.8.
    # Exit when price touches H4/L4 pivot levels or reverses to H6/L6.
    # Uses discrete position sizing (0.25) to target 75-200 trades over 4 years.
    # Works in bull/bear via chop filter avoiding trend-following false signals in ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(5) for Camarilla width
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    
    # Calculate Camarilla levels for each day
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_h6 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    camarilla_l6 = np.zeros_like(close_1d)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    camarilla_h4 = pivot + (1.1/2) * range_hl
    camarilla_h3 = pivot + (1.1/4) * range_hl
    camarilla_h6 = pivot + (1.1/2) * range_hl * 2
    camarilla_l3 = pivot - (1.1/4) * range_hl
    camarilla_l4 = pivot - (1.1/2) * range_hl
    camarilla_l6 = pivot - (1.1/2) * range_hl * 2
    
    # Calculate 1d volume 20-period MA
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d chop regime (using simplified version: ATR(14) / (highest high - lowest low over 14))
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log10(atr_14 * np.sqrt(14) / (hh_14 - ll_14)) / np.log10(10)
    
    # Align 1d indicators to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_spike = volume_1d_aligned[i] > 1.8 * vol_ma_1d_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Price relative to Camarilla levels
        price_above_h3 = close[i] > camarilla_h3_aligned[i]
        price_below_l3 = close[i] < camarilla_l3_aligned[i]
        price_above_h4 = close[i] > camarilla_h4_aligned[i]
        price_below_l4 = close[i] < camarilla_l4_aligned[i]
        price_above_h6 = close[i] > camarilla_h6_aligned[i]
        price_below_l6 = close[i] < camarilla_l6_aligned[i]
        
        # Entry conditions
        if price_above_h3 and volume_spike and chop_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_l3 and volume_spike and chop_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: price reaches H4/L4 or reverses to H6/L6
        elif position == 1 and (price_above_h4 or price_below_l6):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (price_below_l4 or price_above_h6):
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

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0