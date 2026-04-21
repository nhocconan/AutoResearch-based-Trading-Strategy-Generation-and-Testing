#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_ChopRegime_V2
Hypothesis: Camarilla R1/S1 breakouts with volume confirmation and choppiness regime filter work on 12h timeframe for BTC and ETH. Uses 1d for Camarilla levels, 12h EMA50 for trend, and 12h choppiness index to avoid whipsaws in ranging markets. Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for trend and chop filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla pivot point and R1/S1 levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + (prev_high - prev_low) * 1.1 / 12
    s1 = pivot - (prev_high - prev_low) * 1.1 / 12
    
    # Align daily levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    sum_tr = pd.Series(atr_12h * 14).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum_tr / (hh_12h - ll_12h)) / log10(14)
    range_12h = hh_12h - ll_12h
    chop_raw = np.where(range_12h > 0, np.log10(sum_tr / range_12h) / np.log10(14), 0)
    chop = 100 * chop_raw
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (12h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend filter: EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Choppiness regime filter: avoid ranging markets (CHOP > 61.8)
        chop_ok = chop_aligned[i] <= 61.8
        
        if position == 0:
            # Long: price breaks above R1 in uptrend with volume and chop filter
            if uptrend and volume_ok and chop_ok:
                if price > r1_aligned[i]:
                    signals[i] = 0.30
                    position = 1
            # Short: price breaks below S1 in downtrend with volume and chop filter
            elif downtrend and volume_ok and chop_ok:
                if price < s1_aligned[i]:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: price reaches S1 or stoploss
            if price <= s1_aligned[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price reaches R1 or stoploss
            if price >= r1_aligned[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_ChopRegime_V2"
timeframe = "12h"
leverage = 1.0