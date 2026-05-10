#!/usr/bin/env python3
# 4h_ChoppinessIndex_Regime_Breakout
# Hypothesis: In trending regimes (Choppiness Index < 38.2), trade Donchian(20) breakouts with volume confirmation.
# In ranging regimes (Choppiness Index > 61.8), fade the extremes (sell at upper band, buy at lower band).
# Uses 12h EMA50 to filter breakout direction in trending markets. Designed for 20-40 trades/year to avoid fee drag.
# Works in both bull and bear markets by adapting to market regime.

name = "4h_ChoppinessIndex_Regime_Breakout"
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
    
    # Calculate ATR(14) for Donchian and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Donchian Channel (20)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Choppiness Index (14)
    chop = np.full(n, np.nan)
    for i in range(14, n):
        sum_tr = np.nansum(tr[i-13:i+1])
        hh = np.max(high[i-13:i+1])
        ll = np.min(low[i-13:i+1])
        if hh > ll and sum_tr > 0:
            chop[i] = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Get 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(chop[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trending market: Chop < 38.2 -> trade breakouts
            if chop[i] < 38.2:
                # Long breakout with volume confirmation and trend filter
                if close[i] > upper[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown with volume confirmation and trend filter
                elif close[i] < lower[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            # Ranging market: Chop > 61.8 -> fade extremes
            elif chop[i] > 61.8:
                # Sell at upper band (assume reversal)
                if close[i] > upper[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
                # Buy at lower band (assume reversal)
                elif close[i] < lower[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Exit: Price re-enters Donchian channel or stoploss hit
            if close[i] < upper[i] or (i > 0 and low[i] < upper[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters Donchian channel or stoploss hit
            if close[i] > lower[i] or (i > 0 and high[i] > lower[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals