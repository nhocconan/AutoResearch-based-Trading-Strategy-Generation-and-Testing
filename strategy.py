#!/usr/bin/env python3
# 6h_WeeklyPivot_DailyBreakout_TrendFilter
# Hypothesis: Breakout above/below daily close with volume >2x 20-period average and trend filter from 1w EMA200.
# Weekly pivot provides directional bias (price > weekly pivot = bullish bias, < = bearish bias).
# Daily breakout captures momentum; weekly trend filter avoids counter-trend trades.
# Designed for 15-30 trades/year on 6h timeframe with low frequency to minimize fee drag.

name = "6h_WeeklyPivot_DailyBreakout_TrendFilter"
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(200) with proper initialization
    ema_200_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[0:200])
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = (close_1w[i] * 2 + ema_200_1w[i-1] * 198) / 200
    
    # Align 1w EMA to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for weekly pivot and daily breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate weekly pivot from prior week: (PH + PL + PC) / 3
    # where PH = prior week high, PL = prior week low, PC = prior week close
    weekly_high = np.maximum.accumulate(high_1d)
    weekly_low = np.minimum.accumulate(low_1d)
    weekly_close = close_1d
    
    # Shift to get prior week values
    ph = np.roll(weekly_high, 1)
    pl = np.roll(weekly_low, 1)
    pc = np.roll(weekly_close, 1)
    ph[0] = pl[0] = pc[0] = np.nan
    
    weekly_pivot = (ph + pl + pc) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # Daily breakout: close > prior day high (for long) or close < prior day low (for short)
    prior_day_high = np.roll(high_1d, 1)
    prior_day_low = np.roll(low_1d, 1)
    prior_day_high[0] = prior_day_low[0] = np.nan
    
    # Align daily levels to 6h timeframe
    prior_day_high_aligned = align_htf_to_ltf(prices, df_1d, prior_day_high)
    prior_day_low_aligned = align_htf_to_ltf(prices, df_1d, prior_day_low)
    
    # Volume filter: 6h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or \
           np.isnan(prior_day_high_aligned[i]) or np.isnan(prior_day_low_aligned[i]) or \
           np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above prior day high AND volume confirmation AND bullish bias (price > weekly pivot) AND above weekly EMA200
            if (close[i] > prior_day_high_aligned[i] and 
                volume_ratio[i] > 2.0 and 
                close[i] > weekly_pivot_aligned[i] and 
                close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below prior day low AND volume confirmation AND bearish bias (price < weekly pivot) AND below weekly EMA200
            elif (close[i] < prior_day_low_aligned[i] and 
                  volume_ratio[i] > 2.0 and 
                  close[i] < weekly_pivot_aligned[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below prior day low or trend turns bearish
            if close[i] < prior_day_low_aligned[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above prior day high or trend turns bullish
            if close[i] > prior_day_high_aligned[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals