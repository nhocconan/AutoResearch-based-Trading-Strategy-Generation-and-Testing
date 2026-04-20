#!/usr/bin/env python3
# Strategy: 12h_1d_VolumeROC_MomentumBreakout_v1
# Hypothesis: Breakouts above prior 12h high/low with volume rate-of-change confirmation and 1d trend filter.
# Uses 12h price action for entries, filtered by 1d EMA34 trend to avoid counter-trend trades.
# Volume ROC > 1.5 confirms institutional participation. Designed for 15-35 trades/year to minimize fee drag.
# Works in bull markets (trend continuation) and bear markets (mean-reversion bounces off intraday extremes).
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Load 12h data for price, volume, and volatility
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume ROC (rate of change) - 10 period
    vol_roc = ((pd.Series(volume_12h) / pd.Series(volume_12h).shift(10)) - 1).values
    vol_roc_aligned = align_htf_to_ltf(prices, df_12h, vol_roc)
    
    # Prior 12h high/low for breakout levels
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_high[0] = high_12h[0]
    prev_low[0] = low_12h[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_roc_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol_roc_val = vol_roc_aligned[i]
        
        if position == 0:
            # Long: price breaks above prior 12h high, in uptrend (price > EMA34), with volume expansion
            if (price > prev_high[i] and 
                price > ema34_1d_aligned[i] and 
                vol_roc_val > 0.5):  # 50% volume increase vs 10 periods ago
                signals[i] = 0.25
                position = 1
            # Short: price breaks below prior 12h low, in downtrend (price < EMA34), with volume expansion
            elif (price < prev_low[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_roc_val > 0.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below prior 12h low or trend reversal
            if (price < prev_low[i] or 
                price < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above prior 12h high or trend reversal
            if (price > prev_high[i] or 
                price > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_VolumeROC_MomentumBreakout_v1"
timeframe = "12h"
leverage = 1.0