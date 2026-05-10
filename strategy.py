#!/usr/bin/env python3
# 6h_WeeklyPivot_DonchianBreakout_Trend
# Hypothesis: Breakout above/below weekly Donchian channel (20) on 6h, filtered by daily EMA200 trend and volume confirmation (>1.5x average).
# Weekly pivot provides structural support/resistance; daily trend filters for bias; volume confirms breakout strength.
# Designed for 15-30 trades/year on 6h timeframe to avoid fee drag. Works in bull/bear via trend filter.

name = "6h_WeeklyPivot_DonchianBreakout_Trend"
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
    
    # Calculate ATR(20) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(20, n):
        atr[i] = np.nanmean(tr[i-19:i+1])
    
    # Get daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get weekly data for Donchian channel (20 weeks)
    df_1w = get_htf_data(prices, '1w')
    donchian_high_1w = np.full(len(df_1w), np.nan)
    donchian_low_1w = np.full(len(df_1w), np.nan)
    for i in range(20, len(df_1w)):
        donchian_high_1w[i] = np.nanmax(df_1w['high'].values[i-20:i])
        donchian_low_1w[i] = np.nanmin(df_1w['low'].values[i-20:i])
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of daily EMA200 trend
            if close[i] > ema_200_1d_aligned[i]:  # Uptrend
                # Long: Breakout above weekly Donchian high with volume confirmation
                if close[i] > donchian_high_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: Breakout below weekly Donchian low with volume confirmation
                if close[i] < donchian_low_1w_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: Price closes below EMA200 or stoploss hit
            if close[i] < ema_200_1d_aligned[i] or (i > 0 and low[i] < donchian_low_1w_aligned[i] - 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above EMA200 or stoploss hit
            if close[i] > ema_200_1d_aligned[i] or (i > 0 and high[i] > donchian_high_1w_aligned[i] + 2.0 * atr[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals