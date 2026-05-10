#!/usr/bin/env python3
# 12h_PriceChannel_Reversal_Volume_Regime
# Hypothesis: Price reversals from Donchian channels (20) on 12h with volume confirmation and chop regime filter work in both bull and bear markets by capturing mean-reversion moves at extremes. Uses 1d trend filter and 1w volatility regime to avoid false signals. Target: 50-150 total trades over 4 years.

name = "12h_PriceChannel_Reversal_Volume_Regime"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and weekly for regime
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1w ATR for volatility regime (normalize by price)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.inf], tr1])  # first bar has no TR
    atr_1w = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_norm_1w = atr_1w / close_1w
    atr_norm_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_norm_1w, additional_delay_bars=0)
    
    # 12h Donchian channels (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), volume MA (20), EMA50 (50)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i]) or
            np.isnan(atr_norm_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: avoid extreme volatility (use 80th percentile of ATR norm)
        vol_regime = atr_norm_1w_aligned[i] < np.percentile(atr_norm_1w_aligned[:i+1], 80)
        
        # Trend filter: price relative to 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long: price at lower Donchian band in uptrend regime + volume
            if close[i] <= lowest_low[i] and uptrend and vol_regime and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price at upper Donchian band in downtrend regime + volume
            elif close[i] >= highest_high[i] and downtrend and vol_regime and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches middle of channel or trend breaks
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if close[i] >= mid_channel or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches middle of channel or trend breaks
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if close[i] <= mid_channel or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals