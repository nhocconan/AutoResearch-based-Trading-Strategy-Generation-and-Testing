#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day trend filter.
Long when Bollinger Band width < 20th percentile (squeeze), price breaks above upper band, and 1-day EMA50 rising.
Short when Bollinger Band width < 20th percentile (squeeze), price breaks below lower band, and 1-day EMA50 falling.
Exit when price re-enters Bollinger Bands or Bollinger Band width exceeds 80th percentile (squeeze ends).
Uses volatility contraction/expansion for low-frequency, high-conviction trades.
Works in both bull and bear markets by following daily trend while using 4h Bollinger squeeze for entries.
"""

import numpy as np
import pandas as pd
from mtrader import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    
    upper = sma + (bb_std * std)
    lower = sma - (bb_std * std)
    bandwidth = (upper - lower) / sma  # Normalized bandwidth
    
    # Percentile bands for squeeze detection (using 50-period lookback)
    bandwidth_series = pd.Series(bandwidth)
    bw_lower = bandwidth_series.rolling(window=50, min_periods=20).quantile(0.20).values
    bw_upper = bandwidth_series.rolling(window=50, min_periods=20).quantile(0.80).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(bw_lower[i]) or np.isnan(bw_upper[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger squeeze (bandwidth < 20th percentile) AND price breaks above upper band AND 1-day EMA50 rising
            if (bandwidth[i] < bw_lower[i] and 
                close[i] > upper[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze (bandwidth < 20th percentile) AND price breaks below lower band AND 1-day EMA50 falling
            elif (bandwidth[i] < bw_lower[i] and 
                  close[i] < lower[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price re-enters Bollinger Bands OR squeeze ends (bandwidth > 80th percentile)
                if (close[i] < upper[i] and close[i] > lower[i]) or bandwidth[i] > bw_upper[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price re-enters Bollinger Bands OR squeeze ends (bandwidth > 80th percentile)
                if (close[i] < upper[i] and close[i] > lower[i]) or bandwidth[i] > bw_upper[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Bollinger_Squeeze_Breakout_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0