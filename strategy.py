#!/usr/bin/env python3
# 12h_hma_trend_volume_v1
# Hypothesis: 12h strategy using HMA(21) trend filter from 1d HTF for direction, volume confirmation (>1.5x 20-bar average volume), and discrete position sizing (0.25). Enters long when price > HMA with volume confirmation; enters short when price < HMA with volume confirmation. Exits on trend reversal. Uses weekly timeframe only for HTF alignment safety. Target: 12-37 trades/year (50-150 total over 4 years). HMA reduces lag vs SMA/EMA while volume filters weak moves. Works in bull/bear by following established trend with institutional volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_hma_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period = 10 days of 12h bars)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: 1d HMA(21) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate HMA(21) on 1d close
    n_hma = 21
    half_n = n_hma // 2
    sqrt_n = int(np.sqrt(n_hma))
    
    wma_half = pd.Series(close_1d).rolling(window=half_n, min_periods=half_n).mean().values
    wma_full = pd.Series(close_1d).rolling(window=n_hma, min_periods=n_hma).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filters
        uptrend = close[i] > hma_21_1d_aligned[i]
        downtrend = close[i] < hma_21_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: trend turns down
            if not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns up
            if not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for trend continuation with volume confirmation
            if uptrend and volume_confirmed:
                position = 1
                signals[i] = 0.25
            elif downtrend and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals