#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Bandwidth regime filter + 1d Camarilla pivot breakout
# Long when price breaks above R4 AND bandwidth < 20th percentile (low volatility squeeze)
# Short when price breaks below S4 AND bandwidth < 20th percentile (low volatility squeeze)
# Exit when price reverts to VWAP (mean reversion in range) or bandwidth expands > 80th percentile (trend start)
# Uses discrete sizing 0.25 to control fee drag
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Bollinger Bandwidth identifies low volatility regimes conducive to breakouts
# Camarilla R4/S4 are strong breakout levels from 1d timeframe
# VWAP exit provides mean reversion edge in ranging markets

name = "6h_BBW_Camarilla_R4S4_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bandwidth (20, 2) on 6h data
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean()
    dev = close_series.rolling(window=20, min_periods=20).std()
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    bandwidth = ((upper - lower) / basis) * 100
    
    # Calculate 20th and 80th percentiles of bandwidth for regime filtering
    bandwidth_series = pd.Series(bandwidth)
    bw_percentile_20 = bandwidth_series.rolling(window=100, min_periods=100).quantile(0.20)
    bw_percentile_80 = bandwidth_series.rolling(window=100, min_periods=100).quantile(0.80)
    
    # Calculate VWAP for exit signal
    typical_price = (high + low + close) / 3.0
    vwap = (typical_price * volume).cumsum() / volume.cumsum()
    vwap = vwap.replace(to_replace=np.nan, method='ffill').values
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla levels: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bandwidth[i]) or np.isnan(bw_percentile_20[i]) or 
            np.isnan(bw_percentile_80[i]) or np.isnan(vwap[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: breakout in low volatility regime
            # Long: price breaks above R4 AND bandwidth < 20th percentile (squeeze)
            if close[i] > camarilla_r4_aligned[i] and bandwidth[i] < bw_percentile_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND bandwidth < 20th percentile (squeeze)
            elif close[i] < camarilla_s4_aligned[i] and bandwidth[i] < bw_percentile_20[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reverts to VWAP OR bandwidth expands > 80th percentile (trend start)
            if close[i] <= vwap[i] or bandwidth[i] > bw_percentile_80[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reverts to VWAP OR bandwidth expands > 80th percentile (trend start)
            if close[i] >= vwap[i] or bandwidth[i] > bw_percentile_80[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals