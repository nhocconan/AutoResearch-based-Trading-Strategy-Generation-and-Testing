#!/usr/bin/env python3
"""
4h Camarilla Pivot Level Touch + 1d Volume Spike + ADX Trend Filter
Long: Price touches/breaks above Camarilla H3 + 1d volume spike + ADX > 25
Short: Price touches/breaks below Camarilla L3 + 1d volume spike + ADX > 25
Exit: Price crosses Camarilla H4/L4 or ADX drops below 20
Camarilla levels from daily pivot provide institutional support/resistance.
Volume spike confirms institutional participation. ADX ensures trending environment.
Designed for 4h to capture breakouts with institutional validation.
Target: 80-120 total trades over 4 years (20-30/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given OHLC"""
    pivot = (high + low + close) / 3.0
    range_ = high - low
    h4 = pivot + range_ * 1.1 / 2
    h3 = pivot + range_ * 1.1 / 4
    h2 = pivot + range_ * 1.1 / 6
    h1 = pivot + range_ * 1.1 / 12
    l1 = pivot - range_ * 1.1 / 12
    l2 = pivot - range_ * 1.1 / 6
    l3 = pivot - range_ * 1.1 / 4
    l4 = pivot - range_ * 1.1 / 2
    return h4, h3, h2, h1, l1, l2, l3, l4

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    plus_dm = np.diff(high, prepend=high[0])
    minus_dm = np.diff(low, prepend=low[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = np.abs(np.diff(high, prepend=high[0]))
    tr2 = np.abs(np.diff(low, prepend=low[0]))
    tr3 = np.abs(np.diff(close, prepend=close[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # We need to align daily OHLC to 4h bars
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    h4_1d, h3_1d, h2_1d, h1_1d, l1_1d, l2_1d, l3_1d, l4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 4h
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    h2_1d_aligned = align_htf_to_ltf(prices, df_1d, h2_1d)
    h1_1d_aligned = align_htf_to_ltf(prices, df_1d, h1_1d)
    l1_1d_aligned = align_htf_to_ltf(prices, df_1d, l1_1d)
    l2_1d_aligned = align_htf_to_ltf(prices, df_1d, l2_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # 1d volume spike (2x 20-day average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # ADX on 4h for trend filter
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need ADX and Camarilla calculations
    
    for i in range(start_idx, n):
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or
            np.isnan(h4_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above H3 + volume spike + ADX > 25
            if (price > h3_1d_aligned[i] and 
                volume_spike_1d_aligned[i] > 0.5 and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 + volume spike + ADX > 25
            elif (price < l3_1d_aligned[i] and 
                  volume_spike_1d_aligned[i] > 0.5 and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below H4 OR ADX drops below 20
            if (price < h4_1d_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above L4 OR ADX drops below 20
            if (price > l4_1d_aligned[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Touch_1dVolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0