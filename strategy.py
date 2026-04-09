#!/usr/bin/env python3
# 6h_weekly_pivot_volume_confirmation_v1
# Hypothesis: 6h strategy using weekly pivot points for structure, with volume confirmation.
# Long when price breaks above weekly R1 with volume > 1.5x 20-period average.
# Short when price breaks below weekly S1 with volume > 1.5x 20-period average.
# Exit when price returns to weekly pivot point (PP).
# Weekly pivot points calculated from prior week's OHLC: PP=(H+L+C)/3, R1=2*PP-L, S1=2*PP-H.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL to avoid overtrading and fee drag.
# Weekly pivot provides multi-timeframe structure from 1w timeframe.
# Volume confirmation ensures breakouts have conviction.
# Works in both bull and bear markets: captures breakouts in trending markets and mean reversion at extremes in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_volume_confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Multi-timeframe: get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from prior week's OHLC
    # PP = (High + Low + Close) / 3
    # R1 = (2 * PP) - Low
    # S1 = (2 * PP) - High
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = (2 * pp) - low_1w
    s1 = (2 * pp) - high_1w
    
    # Align weekly pivot levels to 6h timeframe (completed weekly bar only)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation
            bullish_breakout = (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]) and volume_confirmed
            bearish_breakout = (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]) and volume_confirmed
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals