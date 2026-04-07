#!/usr/bin/env python3
"""
6h_chaos_fractal_1w_v1
Hypothesis: Chaos fractal dimension (based on Hurst exponent) detects market regime shifts.
In trending markets (Hurst > 0.6), trend-following works; in ranging markets (Hurst < 0.4), mean-reversion works.
Uses weekly Hurst exponent to determine regime, then applies appropriate strategy:
- Trending: 60-bar Donchian breakout with volume confirmation
- Ranging: Fade at Bollinger Bands (20,2) with RSI divergence
Weekly regime filter adapts to changing market conditions, working in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_chaos_fractal_1w_v1"
timeframe = "6h"
leverage = 1.0

def hurst_exponent(series, max_lag=20):
    """Calculate Hurst exponent using R/S analysis."""
    n = len(series)
    if n < max_lag * 2:
        return 0.5
    
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        # Calculate variance of lagged differences
        pp = np.subtract(series[lag:], series[:-lag])
        tau.append(np.std(pp))
    
    if len(tau) < 2:
        return 0.5
    
    # Linear fit in log-log space
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    hurst = poly[0] * 2.0  # Hurst exponent is 2 * slope
    return np.clip(hurst, 0.0, 1.0)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Weekly data for regime detection
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Hurst exponent (regime detector)
    close_1w = df_1w['close'].values
    hurst_values = np.full(len(close_1w), np.nan)
    
    # Calculate rolling Hurst with lookback of 50 weeks
    for i in range(50, len(close_1w)):
        window = close_1w[i-50:i]
        hurst_values[i] = hurst_exponent(window, max_lag=10)
    
    # Align Hurst to 6h timeframe (previous week's value)
    hurst_aligned = align_htf_to_ltf(prices, df_1w, hurst_values)
    
    # Daily data for Bollinger Bands (used in ranging regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2)
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Align Bollinger Bands to 6h
    bb_middle_aligned = align_htf_to_ltf(prices, df_1d, bb_middle)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # RSI for divergence detection (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Donchian channel (60-period for breakouts)
    donchian_high = pd.Series(high).rolling(window=60, min_periods=60).max().values
    donchian_low = pd.Series(low).rolling(window=60, min_periods=60).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if data not available
        if (np.isnan(hurst_aligned[i]) or np.isnan(close[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        hurst = hurst_aligned[i]
        
        if position == 1:  # Long position
            if hurst > 0.5:  # Trending regime - trail with Donchian low
                if close[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
            else:  # Ranging regime - exit at BB middle or RSI overbought
                if (not np.isnan(bb_middle_aligned[i]) and 
                    close[i] > bb_middle_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                elif (not np.isnan(rsi[i]) and rsi[i] > 70):
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if hurst > 0.5:  # Trending regime - trail with Donchian high
                if close[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
            else:  # Ranging regime - exit at BB middle or RSI oversold
                if (not np.isnan(bb_middle_aligned[i]) and 
                    close[i] < bb_middle_aligned[i]):
                    position = 0
                    signals[i] = 0.0
                elif (not np.isnan(rsi[i]) and rsi[i] < 30):
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Regime-based entry logic
            if hurst > 0.6:  # Strong trending regime
                # Donchian breakout with volume confirmation
                if (not np.isnan(donchian_high[i]) and 
                    close[i] > donchian_high[i] and 
                    volume[i] > vol_ma[i] * 1.5):
                    position = 1
                    signals[i] = 0.25
                elif (not np.isnan(donchian_low[i]) and 
                      close[i] < donchian_low[i] and 
                      volume[i] > vol_ma[i] * 1.5):
                    position = -1
                    signals[i] = -0.25
            elif hurst < 0.4:  # Strong ranging regime
                # Mean reversion at Bollinger Bands with RSI divergence
                if (not np.isnan(bb_lower_aligned[i]) and 
                    not np.isnan(rsi[i]) and
                    close[i] < bb_lower_aligned[i] and 
                    rsi[i] < 30 and
                    # Bullish divergence: price makes lower low, RSI makes higher low
                    i >= 2 and close[i] < close[i-1] and close[i-1] < close[i-2] and
                    rsi[i] > rsi[i-1]):
                    position = 1
                    signals[i] = 0.25
                elif (not np.isnan(bb_upper_aligned[i]) and 
                      not np.isnan(rsi[i]) and
                      close[i] > bb_upper_aligned[i] and 
                      rsi[i] > 70 and
                      # Bearish divergence: price makes higher high, RSI makes lower high
                      i >= 2 and close[i] > close[i-1] and close[i-1] > close[i-2] and
                      rsi[i] < rsi[i-1]):
                    position = -1
                    signals[i] = -0.25
    
    return signals