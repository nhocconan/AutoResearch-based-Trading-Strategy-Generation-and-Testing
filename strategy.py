#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_Breakout_Volume_Regime_ATRStop
Hypothesis: 4h Camarilla R1/S1 breakouts with volume confirmation and 1d chop regime filter.
Long when price breaks above R1 with volume > 1.5x 20-period average and chop < 61.8 (trending).
Short when price breaks below S1 with volume confirmation and chop < 61.8.
ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR.
Uses 1d HTF for chop regime calculation. Target: 20-50 trades/year (80-200 total over 4 years).
Works in bull/bear: chop filter adapts to regime, breakouts capture momentum in both directions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for chop regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Chop regime (EHLERS) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(TR)/ (max(high)-min(low))) / log10(N)
    sum_tr = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * (np.log10(sum_tr) - np.log10(max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Camarilla pivot levels from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Using previous day's OHLC
    prev_close = np.roll(close_4h, 1)  # previous bar close
    prev_high = np.roll(high_4h, 1)    # previous bar high
    prev_low = np.roll(low_4h, 1)      # previous bar low
    prev_close[0] = close_4h[0]        # first bar uses current
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    
    camarilla_width = 1.1 * (prev_high - prev_low) / 12
    r1 = prev_close + camarilla_width
    s1 = prev_close - camarilla_width
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_4h > (1.5 * vol_ma)
    
    # ATR (14-period) for stoploss
    tr1_4h = pd.Series(high_4h - low_4h)
    tr2_4h = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3_4h = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h = tr_4h.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) 
            or np.isnan(volume_spike[i]) or np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        # Regime filter: only trade when chop < 61.8 (trending market)
        in_trending_regime = chop_aligned[i] < 61.8
        
        if position == 0 and in_trending_regime:
            # Long: price breaks above R1 with volume confirmation
            if price > r1[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume confirmation
            elif price < s1[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price returns below R1 (failed breakout) or chop becomes too high
            elif price < r1[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price returns above S1 (failed breakdown) or chop becomes too high
            elif price > s1[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_Regime_ATRStop"
timeframe = "4h"
leverage = 1.0