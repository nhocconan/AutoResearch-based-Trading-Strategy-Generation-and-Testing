#!/usr/bin/env python3

"""
Hypothesis: Daily Bollinger Band breakout with weekly trend filter and volume confirmation.
Buy when price breaks above upper BB with bullish weekly trend and volume spike.
Sell when price breaks below lower BB with bearish weekly trend and volume spike.
Uses Bollinger Bands (20,2) for volatility-based breakouts and weekly trend to avoid counter-trend trades.
Target: 10-20 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, window=20, num_std=2):
    """Calculate Bollinger Bands"""
    ma = pd.Series(close).rolling(window=window, min_periods=window).mean()
    std = pd.Series(close).rolling(window=window, min_periods=window).std()
    upper = ma + (num_std * std)
    lower = ma - (num_std * std)
    return upper.values, ma.values, lower.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Weekly trend: bullish when EMA20 > EMA50
    bullish_trend_1w = ema_20_1w > ema_50_1w
    bearish_trend_1w = ema_20_1w < ema_50_1w
    
    # Align weekly trend to daily timeframe
    bullish_aligned = align_htf_to_ltf(prices, df_1w, bullish_trend_1w.astype(float))
    bearish_aligned = align_htf_to_ltf(prices, df_1w, bearish_trend_1w.astype(float))
    
    # Calculate daily Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2)
    
    # Calculate volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(bullish_aligned[i]) or np.isnan(bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB, bullish weekly trend, volume spike
            if (close[i] > bb_upper[i] and 
                bullish_aligned[i] > 0.5 and 
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB, bearish weekly trend, volume spike
            elif (close[i] < bb_lower[i] and 
                  bearish_aligned[i] > 0.5 and 
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle Bollinger Band
            if position == 1:
                if close[i] < bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > bb_middle[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Bollinger_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0