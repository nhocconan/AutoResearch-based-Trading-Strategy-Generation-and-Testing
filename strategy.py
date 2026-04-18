#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime
Hypothesis: Camarilla pivot levels (R1, S1) from 1-day timeframe act as intraday support/resistance.
Price breaking above R1 with volume confirmation and low chop regime indicates bullish momentum.
Price breaking below S1 with volume confirmation and low chop regime indicates bearish momentum.
Chop regime filter avoids whipsaws in sideways markets. Works in both bull and bear markets
by following institutional price levels. Target: 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each 1-day bar
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    typical_range = df_1d['high'] - df_1d['low']
    r1 = df_1d['close'] + typical_range * 1.1 / 12
    s1 = df_1d['close'] - typical_range * 1.1 / 12
    
    # Align to 4h timeframe (wait for 1-day bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Chop regime filter: Chop <= 61.8 (trending market)
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = max_high - min_low
    
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * (np.log10(sum_atr14) - np.log10(range14)) / np.log10(14)
    chop = np.where(np.isnan(chop), 50, chop)  # Neutral if undefined
    chop_filter = chop <= 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ok = vol_confirm[i]
        chop_ok = chop_filter[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and trending market
            if price > r1_val and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and trending market
            elif price < s1_val and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price breaks below S1 or volatility/chop increases
            if price < s1_val or not vol_ok or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above R1 or volatility/chop increases
            if price > r1_val or not vol_ok or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0