#!/usr/bin/env python3
"""
6h_volatility_squeeze_breakout_v1
Hypothesis: In low volatility regimes (Bollinger Band width < 20th percentile), price breaks out of the
6-hour range with high volume, signaling the start of a new trend. Enter long on breakout above the
6-hour high with volume confirmation, short on breakdown below the 6-hour low. Use 1d trend filter
(EMA50) to align with the daily trend and avoid counter-trend trades. Designed for 6h timeframe to
target 15-30 trades/year, minimizing fee drag. Works in both bull and bear markets by capturing
breakouts from consolidation and using the daily trend filter for direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_volatility_squeeze_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) for volatility squeeze
    ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = ma + 2 * std
    lower = ma - 2 * std
    bb_width = upper - lower
    
    # Percentile rank of BB width (lookback 50 periods)
    def percentile_rank(arr, window):
        pr = np.full_like(arr, np.nan)
        for i in range(window, len(arr)):
            window_data = arr[i-window:i]
            if np.all(np.isnan(window_data)):
                pr[i] = np.nan
            else:
                pr[i] = np.sum(window_data < arr[i]) / np.sum(~np.isnan(window_data)) * 100
        return pr
    
    bb_width_pr = percentile_rank(bb_width, 50)
    
    # 6-hour range (highest high, lowest low over 20 periods ~ 5 days)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(bb_width_pr[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility squeeze condition: BB width < 20th percentile
        squeeze = bb_width_pr[i] < 20
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i-1] if i > 0 else False
        breakdown_down = close[i] < lowest_low[i-1] if i > 0 else False
        
        # Volume confirmation: volume > 30-period average
        vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
        vol_confirmed = not np.isnan(vol_ma[i]) and volume[i] > vol_ma[i]
        
        # Daily trend filter
        bullish_trend = ema_50d_aligned[i] > ema_50d_aligned[i-1] if i > 0 else False
        bearish_trend = ema_50d_aligned[i] < ema_50d_aligned[i-1] if i > 0 else False
        
        if position == 1:  # Long position
            # Exit: price closes below 6-hour VWAP or volatility expands (breakout failed)
            vwap = (pd.Series(close).rolling(window=10).apply(lambda x: np.average(x, weights=np.ones(len(x))), raw=True)).values
            if not np.isnan(vwap[i]) and close[i] < vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6-hour VWAP or volatility expands
            vwap = (pd.Series(close).rolling(window=10).apply(lambda x: np.average(x, weights=np.ones(len(x))), raw=True)).values
            if not np.isnan(vwap[i]) and close[i] > vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: volatility squeeze + breakout up + volume confirmed + bullish daily trend
            if squeeze and breakout_up and vol_confirmed and bullish_trend:
                position = 1
                signals[i] = 0.25
            # Short: volatility squeeze + breakdown down + volume confirmed + bearish daily trend
            elif squeeze and breakdown_down and vol_confirmed and bearish_trend:
                position = -1
                signals[i] = -0.25
    
    return signals