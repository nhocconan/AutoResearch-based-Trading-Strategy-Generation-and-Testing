#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze + Volume Spike + 1d Close Trend
# Uses Bollinger Band width percentile to detect low volatility squeezes,
# breaks out with volume confirmation, and follows 1d close trend for direction.
# Designed to work in both bull and bear markets by capturing volatility breakouts.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for squeeze detection
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_width = (upper - lower) / sma  # Normalized width
    
    # Bollinger Band width percentile (50-period) to identify squeezes
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 1-day close trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_1d = pd.Series(close_1d).rolling(window=10, min_periods=10).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(sma_1d_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Squeeze condition: BB width in lowest 20% percentile
        is_squeeze = bb_width_percentile[i] <= 20
        
        # Breakout condition: price outside Bollinger Bands
        breakout_up = close[i] > upper[i]
        breakout_down = close[i] < lower[i]
        
        # Direction from 1d SMA: above = bullish bias, below = bearish bias
        bullish_bias = close[i] > sma_1d_aligned[i]
        bearish_bias = close[i] < sma_1d_aligned[i]
        
        # Long: squeeze breakout up + volume + bullish bias
        if (is_squeeze and breakout_up and volume[i] > vol_threshold[i] and bullish_bias):
            signals[i] = 0.25
        
        # Short: squeeze breakout down + volume + bearish bias
        elif (is_squeeze and breakout_down and volume[i] > vol_threshold[i] and bearish_bias):
            signals[i] = -0.25
        
        # Exit: volatility expands or price returns to mean
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (bb_width_percentile[i] > 40 or close[i] < sma[i])) or
               (signals[i-1] == -0.25 and (bb_width_percentile[i] > 40 or close[i] > sma[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_BB_Squeeze_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0