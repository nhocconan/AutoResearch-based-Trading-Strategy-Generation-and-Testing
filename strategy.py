#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Camarilla pivot breakout with 1w volume confirmation and chop regime filter.
    # Long when price breaks above Camarilla H3 AND 1w volume > 1.3x 20-period MA AND chop > 61.8 (range regime).
    # Short when price breaks below Camarilla L3 AND 1w volume > 1.3x 20-period MA AND chop > 61.8.
    # Exit when price returns to Camarilla pivot point (mean of H3/L3).
    # Uses discrete position sizing (0.25) to target 30-100 trades over 4 years.
    # Works in bull/bear via chop filter avoiding trend-following false signals in ranging markets.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1w volume 20-period MA
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate True Range for chop calculation
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate 1w chop regime: ATR(14) / (highest high - lowest low over 14) * 100 * log10(sqrt(14))/log10(10)
    hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    range_14 = hh_14 - ll_14
    chop = np.where(range_14 > 0, 100 * np.log10(atr_14 * np.sqrt(14) / range_14) / np.log10(10), 50)
    
    # Calculate 1w Camarilla pivot levels (H3, L3, pivot)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4, Pivot = (high+low+close)/3
    camarilla_h3 = close_1w + 1.1 * (high_1w - low_1w) / 4
    camarilla_l3 = close_1w - 1.1 * (high_1w - low_1w) / 4
    camarilla_pivot = (high_1w + low_1w + close_1w) / 3
    
    # Align 1w indicators to 1d timeframe
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.3x 20-period average
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_spike = volume_1w_aligned[i] > 1.3 * vol_ma_1w_aligned[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        chop_filter = chop_aligned[i] > 61.8
        
        # Camarilla breakout conditions
        breakout_long = close[i] > camarilla_h3_aligned[i-1]  # Break above previous period H3
        breakout_short = close[i] < camarilla_l3_aligned[i-1]  # Break below previous period L3
        
        # Exit conditions: price returns to Camarilla pivot point
        exit_long = close[i] < camarilla_pivot_aligned[i]
        exit_short = close[i] > camarilla_pivot_aligned[i]
        
        # Entry conditions
        if breakout_long and volume_spike and chop_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and volume_spike and chop_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0