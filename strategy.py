#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_RegimeFilter_V1
Hypothesis: Camarilla R1/S1 breakout with volume confirmation and choppiness regime filter (CHOP > 61.8 = range) works on 12h for BTC and ETH in both bull and bear markets. Uses 1d for Camarilla calculation. Tight entries to avoid overtrading. Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 1d timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar: use same values (will be filtered by min_periods later)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + range_1d * 1.1 / 12
    camarilla_s1 = prev_close - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume filter: 20-period average on 12h timeframe
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (14-period) on 12h timeframe
    # CHOP > 61.8 = ranging market (good for mean reversion at pivots)
    # CHOP < 38.2 = trending market (avoid false breakouts)
    hl14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / np.log(14) / (hl14 - ll14))
    chop = np.where((hl14 - ll14) == 0, 50, chop)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average to reduce trades)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        regime_ok = chop[i] > 61.8
        
        if position == 0:
            # Long: price breaks above Camarilla R1 with volume and in ranging regime
            if price > camarilla_r1_aligned[i]:
                if volume_ok and regime_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Camarilla S1 with volume and in ranging regime
            elif price < camarilla_s1_aligned[i]:
                if volume_ok and regime_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price closes below Camarilla S1 (mean reversion) or ATR stoploss
            if price < camarilla_s1_aligned[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Camarilla R1 (mean reversion) or ATR stoploss
            if price > camarilla_r1_aligned[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_RegimeFilter_V1"
timeframe = "12h"
leverage = 1.0