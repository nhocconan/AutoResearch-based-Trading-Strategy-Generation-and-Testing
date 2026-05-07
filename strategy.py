#!/usr/bin/env python3
name = "1d_Weekly_Channel_Breakout_With_Volume_Filter"
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
    
    # Weekly Donchian channels (20-week lookback)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Weekly volume confirmation: current week volume > 1.5x 4-week average
    vol_ma_4w = pd.Series(df_1w['volume']).rolling(window=4, min_periods=4).mean().values
    vol_ma_4w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_4w)
    vol_ratio = df_1w['volume'].values / vol_ma_4w_aligned
    vol_ratio = np.where(np.isnan(vol_ratio) | (vol_ma_4w_aligned == 0), 0, vol_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for weekly Donchian
    
    for i in range(start_idx, n):
        if np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly high with volume confirmation
            if close[i] > high_20w_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly low with volume confirmation
            elif close[i] < low_20w_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close back below weekly midpoint or volume drops
            weekly_mid = (high_20w_aligned[i] + low_20w_aligned[i]) / 2
            if close[i] < weekly_mid or vol_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close back above weekly midpoint or volume drops
            weekly_mid = (high_20w_aligned[i] + low_20w_aligned[i]) / 2
            if close[i] > weekly_mid or vol_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly Donchian breakout with volume confirmation.
# Long when price breaks above 20-week high with volume > 1.5x 4-week average.
# Short when price breaks below 20-week low with volume > 1.5x 4-week average.
# Exit when price returns to weekly midpoint or volume normalizes.
# Weekly timeframe reduces noise and false breakouts; volume filter ensures conviction.
# Works in both bull (breakouts up) and bear (breakouts down) markets.
# Target: 10-25 trades/year to minimize fee decay while capturing major moves.