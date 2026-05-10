#!/usr/bin/env python3
# 4h_TRIX_Volume_Spike_Regime
# Hypothesis: TRIX momentum with volume spike confirmation and chop regime filter.
# Long when TRIX crosses above signal line with volume spike in trending market (CHOP < 38.2).
# Short when TRIX crosses below signal line with volume spike in trending market.
# Uses 1d trend for bias, 4h for entry. Designed to work in both bull and bear markets by
# filtering for trending regimes and requiring volume confirmation to avoid false signals.
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_TRIX_Volume_Spike_Regime"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d_up = close_1d > ema34_1d
    trend_1d_down = close_1d < ema34_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 4h data for TRIX and chop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # TRIX: triple EMA of close
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix_signal = trix.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_hist = trix - trix_signal
    
    # Align TRIX and signal to 4h
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix.values)
    trix_signal_aligned = align_htf_to_ltf(prices, df_4h, trix_signal.values)
    trix_hist_aligned = align_htf_to_ltf(prices, df_4h, trix_hist.values)
    
    # Chop regime: CHOP(14)
    atr1 = np.maximum(high_4h - low_4h,
                      np.maximum(np.abs(high_4h - np.roll(close_4h, 1)),
                                 np.abs(low_4h - np.roll(close_4h, 1))))
    atr1[0] = high_4h[0] - low_4h[0]
    atr_sum = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or
            np.isnan(trix_hist_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_filter = vol_ratio > 1.5
        trending = chop_aligned[i] < 38.2
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume spike in trending market
            if (trix_hist_aligned[i] > 0 and trix_hist_aligned[i-1] <= 0 and
                volume_filter and trending and trend_1d_up_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line with volume spike in trending market
            elif (trix_hist_aligned[i] < 0 and trix_hist_aligned[i-1] >= 0 and
                  volume_filter and trending and trend_1d_down_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below signal line or trend fails
            if (trix_hist_aligned[i] < 0 or
                trend_1d_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: TRIX crosses above signal line or trend fails
            if (trix_hist_aligned[i] > 0 or
                trend_1d_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals