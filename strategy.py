#!/usr/bin/env python3
name = "1d_1w_WeeklyTrend_With_DailyVolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    ema_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Daily volume filter: > 2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA with volume confirmation
            if close[i] > ema_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: Price below weekly EMA with volume confirmation
            elif close[i] < ema_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: Price crosses below weekly EMA
            if close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price crosses above weekly EMA
            if close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Weekly EMA(20) trend filter with daily volume confirmation on 1d timeframe.
# Weekly EMA provides robust trend direction resistant to daily noise.
# Volume filter (2.0x 20-day average) ensures institutional participation and reduces false signals.
# Position size 0.30 balances return and drawdown. Expected trades: 15-25/year to minimize fee drag.
# Works in bull markets (riding uptrends) and bear markets (selling rallies in downtrends).