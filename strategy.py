#!/usr/bin/env python3
# 4h_Choppiness_Donchian_Breakout_12hTrend
# Hypothesis: Use Donchian(20) breakout on 4h with 12h EMA trend filter and Choppiness Index regime filter.
# Enter long when price breaks above Donchian upper band with volume, in uptrend and trending market (CHOP < 38.2).
# Enter short when price breaks below Donchian lower band with volume, in downtrend and trending market.
# Exit when price returns to 20-period EMA or trend reverses.
# Uses 4h for signal, 12h for trend filter, and 4h for Choppiness Index.
# Designed for low frequency (20-40 trades/year) by using multiple filters to reduce false signals.

name = "4h_Choppiness_Donchian_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0

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
    
    # === 4h data for Donchian Channel and Choppiness Index ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian Channel (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA(20) for exit
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10) for Choppiness Index
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    
    # Align 4h indicators to LTF (wait for 4h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # === 12h data for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(50) on 12h for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # Regime filter: trending market (CHOP < 38.2)
        trending_market = chop_aligned[i] < 38.2
        
        if position == 0:
            # LONG: Price breaks above Donchian high with volume, in uptrend and trending market
            if close[i] > donchian_high_aligned[i] and vol_ok and trend_up and trending_market:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low with volume, in downtrend and trending market
            elif close[i] < donchian_low_aligned[i] and vol_ok and trend_down and trending_market:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to EMA20 or trend reverses or market becomes ranging
            if close[i] <= ema_20_4h_aligned[i] or not trend_up or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to EMA20 or trend reverses or market becomes ranging
            if close[i] >= ema_20_4h_aligned[i] or not trend_down or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals