#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h trend and daily volatility contraction breakout.
# Uses 12h EMA20 for trend direction and daily Bollinger Band squeeze (low volatility) for entry timing.
# Breakouts occur when price breaks above/below the 6h high/low of the prior 6-bar period
# during low volatility regimes, filtered by 12h trend alignment.
# Designed to capture explosive moves after consolidation in both bull and bear markets.
# Target: 15-35 trades per year to minimize fee drag.
name = "6h_BollingerSqueeze_12hEMA20_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA20 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA20
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Get daily data for Bollinger Bands (volatility contraction)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Bollinger Bands (20, 2)
    daily_close = df_1d['close'].values
    sma_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(daily_close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized bandwidth
    
    # Align daily Bollinger width to 6h
    bb_width_6h = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 6-bar high/low for breakout levels (prior 6 periods)
    high_6bar = np.full(n, np.nan)
    low_6bar = np.full(n, np.nan)
    for i in range(6, n):
        high_6bar[i] = np.max(high[i-6:i])
        low_6bar[i] = np.min(low[i-6:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for EMA20 and Bollinger Bands
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_6h[i]) or np.isnan(bb_width_6h[i]) or 
            np.isnan(high_6bar[i]) or np.isnan(low_6bar[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility squeeze: Bollinger Band width below 20th percentile of last 50 days
        if i >= 50:
            bb_width_hist = bb_width_6h[max(0, i-50):i]
            bb_width_percentile = (bb_width_6h[i] <= bb_width_hist).mean() * 100
            vol_squeeze = bb_width_percentile <= 20  # Low volatility regime
        else:
            vol_squeeze = False
        
        if position == 0:
            # Long: price breaks above 6-bar high, 12h uptrend (close > EMA20), volatility squeeze
            if (close[i] > high_6bar[i] and 
                close[i] > ema_20_6h[i] and 
                vol_squeeze):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 6-bar low, 12h downtrend (close < EMA20), volatility squeeze
            elif (close[i] < low_6bar[i] and 
                  close[i] < ema_20_6h[i] and 
                  vol_squeeze):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 6-bar low or trend reversal
            if close[i] < low_6bar[i] or close[i] < ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 6-bar high or trend reversal
            if close[i] > high_6bar[i] or close[i] > ema_20_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals