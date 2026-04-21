#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter_V1
Hypothesis: Camarilla R1/S1 breakouts from 1d pivots with 4h EMA50 trend filter and choppiness regime (CHOP<61.8) 
captures high-probability directional moves in both bull and bear markets. Volume confirmation (>1.2x 20-bar average) 
filters weak breakouts. ATR(14) stoploss at 2.0x ATR. Designed for moderate trade frequency (target: 20-40 trades/year) 
to balance opportunity with fee drag minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots, trend, chop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d EMA50 for HTF trend filter ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Choppiness Index regime filter (CHOP < 61.8 = trending) ===
    chop_period = 14
    atr_1d = []
    for i in range(len(high_1d)):
        if i == 0:
            tr = high_1d[i] - low_1d[i]
        else:
            tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        atr_1d.append(tr)
    atr_1d = np.array(atr_1d)
    atr_ma = pd.Series(atr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    high_low_range = pd.Series(high_1d - low_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    chop = 100 * np.log10(high_low_range / atr_ma) / np.log10(chop_period)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Indicators (primary timeframe) ===
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Volume confirmation: current volume > 1.2x 20-period average
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.2 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_aligned[i])
            or np.isnan(volume_threshold[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + HTF uptrend + trending regime
            if price > r1_1d_aligned[i] and volume_4h[i] > volume_threshold[i] and price > ema_50_1d_aligned[i] and chop_aligned[i] < 61.8:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + HTF downtrend + trending regime
            elif price < s1_1d_aligned[i] and volume_4h[i] > volume_threshold[i] and price < ema_50_1d_aligned[i] and chop_aligned[i] < 61.8:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes back below R1 (breakout failed)
            elif price < r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes back above S1 (breakout failed)
            elif price > s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_HTFTrend_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0