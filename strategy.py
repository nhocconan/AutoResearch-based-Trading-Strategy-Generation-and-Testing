#!/usr/bin/env python3
# 4h_Choppiness_Adjusted_Donchian_Breakout_Volume
# Hypothesis: 4h Donchian breakout (20-period) filtered by 12h choppiness regime (Choppiness Index > 61.8 = range-bound, < 38.2 = trending).
# In trending regimes (CHOP < 38.2), we take breakouts in the direction of the 12h EMA trend.
# Volume confirmation (2x 24-period MA) filters weak breakouts. Designed to avoid false breakouts in chop.
# Works in bull/bear by using trend filter + regime filter to avoid whipsaws. Target: 20-50 trades/year.

name = "4h_Choppiness_Adjusted_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 12h data for trend and choppiness
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr = np.concatenate([[np.nan], tr])  # align to same length
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(ATR)/ (HH - LL)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.full_like(close_12h, np.nan)
    mask = (range_hl > 0) & ~np.isnan(sum_atr)
    chop[mask] = 100 * np.log10(sum_atr[mask] / range_hl[mask]) / np.log10(14)
    
    # Trend: 12h close > EMA50
    uptrend_12h = close_12h > ema_50_12h
    downtrend_12h = close_12h < ema_50_12h
    
    # Align 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h.astype(float))
    downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h.astype(float))
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (24-period for 4h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for 12h EMA50 (50) and Donchian (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when market is trending (CHOP < 38.2)
        if chop_aligned[i] >= 38.2:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Donchian high in uptrend with volume surge
            if close[i] > donchian_high[i] and uptrend_12h_aligned[i] > 0.5 and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low in downtrend with volume surge
            elif close[i] < donchian_low[i] and downtrend_12h_aligned[i] > 0.5 and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below Donchian low or trend fails
                if close[i] < donchian_low[i] or uptrend_12h_aligned[i] <= 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above Donchian high or trend fails
                if close[i] > donchian_high[i] or downtrend_12h_aligned[i] <= 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals