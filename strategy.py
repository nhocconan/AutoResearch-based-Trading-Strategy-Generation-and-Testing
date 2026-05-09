#!/usr/bin/env python3
# 6h_IBS_Volume_Trend_1wTrend
# Hypothesis: 6h IBS (Internal Bar Strength) with volume confirmation and weekly trend filter.
# Long when: weekly trend up, IBS < 0.3 (oversold), volume > 1.3x average.
# Short when: weekly trend down, IBS > 0.7 (overbought), volume > 1.3x average.
# Exit when IBS reverses (long: IBS > 0.7, short: IBS < 0.3).
# IBS = (close - low) / (high - low) measures intraday position; effective mean reversion tool.
# Weekly trend filter avoids counter-trend trades in strong trends.
# Target: 60-120 total trades over 4 years (15-30/year) with controlled turnover.

name = "6h_IBS_Volume_Trend_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA20 for trend filter
    ema20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema20_1w[i] = (close_1w[i] * 2 + ema20_1w[i-1] * 18) / 20
    
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate IBS: (close - low) / (high - low)
    ibs = np.zeros_like(close)
    hl_range = high - low
    valid_hl = hl_range != 0
    ibs[valid_hl] = (close[valid_hl] - low[valid_hl]) / hl_range[valid_hl]
    
    # Volume filter: current volume vs 20-period average
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
    
    start_idx = max(20, 20)  # Need weekly EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(ibs[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema20_1w_aligned[i]
        
        if position == 0:
            # Enter long: weekly trend up + IBS oversold + volume confirmation
            if trend_up and ibs[i] < 0.3 and volume_ratio[i] > 1.3:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly trend down + IBS overbought + volume confirmation
            elif not trend_up and ibs[i] > 0.7 and volume_ratio[i] > 1.3:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: IBS overbought (mean reversion complete)
            if ibs[i] > 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: IBS oversold (mean reversion complete)
            if ibs[i] < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals