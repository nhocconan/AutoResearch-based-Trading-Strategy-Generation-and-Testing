#!/usr/bin/env python3
# 6E_WeeklyPivot_DonchianBreakout_Trend
# Hypothesis: Use weekly pivot points (from 1w) to establish directional bias, then enter on 6h Donchian(20) breakout with volume confirmation. Weekly pivot acts as a regime filter (above/below weekly pivot = long/short bias). Works in bull/bear because the pivot adapts to price levels and the breakout captures momentum in the direction of the higher-timeframe structure.

name = "6E_WeeklyPivot_DonchianBreakout_Trend"
timeframe = "6h"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Get 1w data for weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    # Support/resistance levels: R1 = 2*P - L, S1 = 2*P - H, R2 = P + (H - L), S2 = P - (H - L)
    # We'll use the pivot point itself as the bias filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Align weekly pivot to 6h timeframe (with 1-week delay for completed bar)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Donchian channel (20) on 6h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.nanmax(high[i-20:i])
        donchian_low[i] = np.nanmin(low[i-20:i])
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine bias from weekly pivot: above pivot = long bias, below = short bias
            if close[i] > weekly_pivot_aligned[i]:  # Above weekly pivot -> long bias
                # Long: Breakout above Donchian high with volume confirmation
                if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Below weekly pivot -> short bias
                # Short: Breakout below Donchian low with volume confirmation
                if close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below weekly pivot or stoploss hit
            if close[i] < weekly_pivot_aligned[i] or (i > 0 and low[i] < donchian_low[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above weekly pivot or stoploss hit
            if close[i] > weekly_pivot_aligned[i] or (i > 0 and high[i] > donchian_high[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals