#!/usr/bin/env python3
name = "6h_BollingerBands_WaveTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate WaveTrend on 6h (similar to TSI)
    # EMA1 of close
    ema1 = np.full(n, np.nan)
    ema1[0] = close[0]
    for i in range(1, n):
        ema1[i] = 0.1 * close[i] + 0.9 * ema1[i-1]
    
    # EMA2 of EMA1
    ema2 = np.full(n, np.nan)
    ema2[0] = ema1[0]
    for i in range(1, n):
        ema2[i] = 0.1 * ema1[i] + 0.9 * ema2[i-1]
    
    # EMA1 of (close - ema2)
    diff = close - ema2
    ema_diff = np.full(n, np.nan)
    ema_diff[0] = diff[0]
    for i in range(1, n):
        ema_diff[i] = 0.1 * diff[i] + 0.9 * ema_diff[i-1]
    
    # EMA2 of ema_diff
    ema2_diff = np.full(n, np.nan)
    ema2_diff[0] = ema_diff[0]
    for i in range(1, n):
        ema2_diff[i] = 0.1 * ema_diff[i] + 0.9 * ema2_diff[i-1]
    
    # WaveTrend WT1 and WT2
    wt1 = np.full(n, np.nan)
    wt2 = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(ema2_diff[i]) and ema2_diff[i] != 0:
            wt1[i] = ema_diff[i] / ema2_diff[i] * 60
        else:
            wt1[i] = 0
        wt2[i] = np.mean(wt1[max(0, i-3):i+1]) if i >= 3 else wt1[i]
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # ~12 hours for 6h
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        if np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or np.isnan(wt1[i]) or np.isnan(wt2[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine if price is near Bollinger Bands
        near_upper = close[i] >= upper_bb_aligned[i] * 0.995  # within 0.5% of upper band
        near_lower = close[i] <= lower_bb_aligned[i] * 1.005  # within 0.5% of lower band
        
        # WaveTrend crossover signals
        wt_cross_up = wt1[i-1] < wt2[i-1] and wt1[i] >= wt2[i]
        wt_cross_down = wt1[i-1] > wt2[i-1] and wt1[i] <= wt2[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price near lower BB + WT bullish cross + volume
            if near_lower and wt_cross_up and vol_filter[i]:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price near upper BB + WT bearish cross + volume
            elif near_upper and wt_cross_down and vol_filter[i]:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price near upper BB or WT bearish cross
            if near_upper or wt_cross_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price near lower BB or WT bullish cross
            if near_lower or wt_cross_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Bollinger Bands (20,2) from daily timeframe combined with WaveTrend oscillator on 6h.
# Long when price touches lower Bollinger Band with WaveTrend bullish crossover and volume confirmation.
# Short when price touches upper Bollinger Band with WaveTrend bearish crossover and volume confirmation.
# Uses Bollinger Bands for dynamic support/resistance and WaveTrend for momentum timing.
# Works in ranging markets (BB reversals) and trending markets (WT crossovers with BB context).
# Volume filter ensures participation. Target: 50-150 trades over 4 years (12-37/year).