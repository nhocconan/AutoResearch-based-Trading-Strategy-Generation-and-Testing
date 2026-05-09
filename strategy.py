#!/usr/bin/env python3
# Hypothesis: 12h timeframe with weekly ADX regime filter and daily RSI mean reversion.
# In low volatility/trending regime (weekly ADX < 25), price tends to mean-revert to the daily RSI mean (50).
# Enters long when daily RSI < 30 (oversold) and weekly ADX < 25, short when daily RSI > 70 (overbought) and weekly ADX < 25.
# Uses daily volume confirmation: only take trades when volume > 1.5x 20-day average volume.
# Exits when RSI returns to 50 or weekly ADX rises above 25 (trend strengthens).
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WeeklyADX_DailyRSI_MeanReversion"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly ADX (14-period) for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high']
    low_1w = df_1w['low']
    close_1w = df_1w['close']
    
    # Calculate ADX components
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    
    tr = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.abs(high_1w[1:] - close_1w[:-1]),
        np.abs(low_1w[1:] - close_1w[:-1])
    )
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.sum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    if len(plus_dm) < 14 or len(tr) < 14:
        return np.zeros(n)
        
    plus_di_14 = 100 * wilder_smooth(plus_dm, 14) / wilder_smooth(tr, 14)
    minus_di_14 = 100 * wilder_smooth(minus_dm, 14) / wilder_smooth(tr, 14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = wilder_smooth(dx, 14)
    
    # Handle NaN values in ADX calculation
    adx_14 = np.where(np.isnan(adx_14), 0, adx_14)
    adx_14 = np.concatenate([np.full(14, np.nan), adx_14])
    
    adx_values = adx_14
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    # Calculate daily RSI (14-period) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    delta = close_1d.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Neutral RSI when no data
    
    rsi_values = rsi.values
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    volume_1d = df_1d['volume']
    avg_volume_20 = volume_1d.rolling(window=20, min_periods=20).mean()
    volume_surge = volume_1d > (1.5 * avg_volume_20)
    volume_surge_values = volume_surge.values
    volume_surge_aligned = align_htf_to_ltf(prices, df_1d, volume_surge_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(volume_surge_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: oversold RSI + low ADX (ranging market) + volume confirmation
            if (rsi_aligned[i] < 30 and 
                adx_aligned[i] < 25 and 
                volume_surge_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: overbought RSI + low ADX (ranging market) + volume confirmation
            elif (rsi_aligned[i] > 70 and 
                  adx_aligned[i] < 25 and 
                  volume_surge_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral OR ADX rises (trend developing)
            if (rsi_aligned[i] >= 50) or (adx_aligned[i] >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral OR ADX rises (trend developing)
            if (rsi_aligned[i] <= 50) or (adx_aligned[i] >= 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals