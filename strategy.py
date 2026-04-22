#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Bandwidth squeeze breakout with 1-day trend filter and volume confirmation.
Long when price breaks above upper BB during low volatility (BW < 20th percentile) and 1d EMA50 up.
Short when price breaks below lower BB during low volatility and 1d EMA50 down.
Exit when price reverts to middle BB or volatility expands (BW > 80th percentile).
Works in both bull and bear markets by following the 1d trend during low-volatility breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands: 20-period, 2 std dev
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Bandwidth: (upper - lower) / basis
    bw = (upper_band - lower_band) / basis
    # Percentile lookback: 50 periods
    bw_lower = pd.Series(bw).rolling(window=50, min_periods=50).quantile(0.20).values
    bw_upper = pd.Series(bw).rolling(window=50, min_periods=50).quantile(0.80).values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bw[i]) or np.isnan(bw_lower[i]) or 
            np.isnan(bw_upper[i]) or np.isnan(vol_ma_30[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(basis[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility and breakout conditions
        low_volatility = bw[i] < bw_lower[i]
        high_volatility = bw[i] > bw_upper[i]
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        vol_confirm = volume[i] > 1.5 * vol_ma_30[i]
        
        if position == 0:
            # Long: breakout up during low vol, 1d EMA50 up, volume confirmation
            if low_volatility and breakout_up and ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout down during low vol, 1d EMA50 down, volume confirmation
            elif low_volatility and breakout_down and ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to middle band or volatility expands
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= middle band or volatility expands
                if close[i] <= basis[i] or high_volatility:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price >= middle band or volatility expands
                if close[i] >= basis[i] or high_volatility:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_BollingerBandwidthSqueeze_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0