#!/usr/bin/env python3
# 1d_Williams_Alligator_Strategy
# Hypothesis: Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) identifies trends.
# Price above all lines = uptrend, below all = downtrend. Adds weekly trend filter (EMA50)
# and volume confirmation (>1.5x 20-period average volume) to avoid whipsaws.
# Works in bull/bear markets: Alligator catches trends, volume confirms strength,
# weekly trend ensures alignment with higher timeframe momentum.
# Target: 15-25 trades/year.

name = "1d_Williams_Alligator_Strategy"
timeframe = "1d"
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
    
    # Williams Alligator SMAs
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # Red line (8)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # Green line (5)
    
    # Alligator alignment: all lines in order
    alligator_long = (lips > teeth) & (teeth > jaw)   # Green > Red > Blue = uptrend
    alligator_short = (lips < teeth) & (teeth < jaw)  # Green < Red < Blue = downtrend
    
    # Weekly trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to daily
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned up + weekly uptrend + volume confirmation
            if (alligator_long[i] and
                trend_1w_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + weekly downtrend + volume confirmation
            elif (alligator_short[i] and
                  trend_1w_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Alligator alignment breaks or weekly trend turns down
            if (not alligator_long[i] or
                trend_1w_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Alligator alignment breaks or weekly trend turns up
            if (not alligator_short[i] or
                trend_1w_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals