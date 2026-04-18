#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume and ADX Filter
Hypothesis: Camarilla pivot levels (especially H3/L3) act as strong support/resistance. A breakout above H3 or below L3 with volume confirmation and ADX > 25 indicates a strong trend. This strategy targets trending moves while avoiding whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d data
    # Camarilla formulas:
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.55*(high-low)
    # H1 = close + 0.275*(high-low)
    # L1 = close - 0.275*(high-low)
    # L2 = close - 0.55*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    H3 = close_1d + 1.1 * (high_1d - low_1d)
    L3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align H3 and L3 to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Calculate ADX on 4h data for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder smoothing (using alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        # Initial value: simple average
        result[period-1] = np.mean(data[:period])
        # Wilder smoothing: new = (old * (period-1) + new) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smooth(tr, 14)
    plus_di = 100 * wilders_smooth(plus_dm, 14) / np.where(atr != 0, atr, 1)
    minus_di = 100 * wilders_smooth(minus_dm, 14) / np.where(atr != 0, atr, 1)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smooth(dx, 14)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[max(0, i-19):i+1])
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or np.isnan(adx[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        h3_val = H3_aligned[i]
        l3_val = L3_aligned[i]
        adx_val = adx[i]
        vol_ok = vol_spike[i]
        
        if position == 0:
            # Enter long: price breaks above H3 with volume and trend
            if (close[i] > h3_val and 
                adx_val > 25 and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below L3 with volume and trend
            elif (close[i] < l3_val and 
                  adx_val > 25 and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below H3 or trend weakens
            if close[i] < h3_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above L3 or trend weakens
            if close[i] > l3_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_Volume_ADX"
timeframe = "4h"
leverage = 1.0