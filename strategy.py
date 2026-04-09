#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot long breakout + 4h HMA(21) trend filter + volume confirmation
# Camarilla pivot provides intraday support/resistance levels for breakout entries
# 4h HMA confirms higher timeframe trend direction to avoid counter-trend trades
# Volume ensures breakout authenticity; discrete sizing 0.20 controls drawdown
# Session filter (08-20 UTC) reduces noise trades outside active hours
# Works in bull/bear: trend filter adapts, breakouts work in both directions
# Target: 60-150 total trades over 4 years = 15-37/year for 1h

name = "1h_4h_camarilla_hma_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h HMA(21)
    close_4h = df_4h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        wma_vals = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            wma_vals[i] = np.dot(values[i - window + 1:i + 1], weights) / weights.sum()
        return wma_vals
    
    wma_half = wma(close_4h, half_len)
    wma_full = wma(close_4h, 21)
    hma_4h = 2 * wma_half - wma_full
    hma_4h = wma(hma_4h, sqrt_len)
    
    # Align 4h HMA to 1h timeframe (wait for 4h bar close)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Load 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h5 = np.full(len(df_1d), np.nan)  # Resistance level
    camarilla_l5 = np.full(len(df_1d), np.nan)  # Support level
    camarilla_h3 = np.full(len(df_1d), np.nan)  # Resistance level
    camarilla_l3 = np.full(len(df_1d), np.nan)  # Support level
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_h5[i] = np.nan
            camarilla_l5[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
        else:
            # Camarilla formulas using previous day's OHLC
            hlc3 = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
            range_1d = high_1d[i-1] - low_1d[i-1]
            
            camarilla_h5[i] = close_1d[i-1] + range_1d * 1.1 / 2
            camarilla_l5[i] = close_1d[i-1] - range_1d * 1.1 / 2
            camarilla_h3[i] = close_1d[i-1] + range_1d * 1.1 / 4
            camarilla_l3[i] = close_1d[i-1] - range_1d * 1.1 / 4
    
    # Align 1d Camarilla levels to 1h timeframe (wait for 1d bar close)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(hma_4h_aligned[i]) or np.isnan(avg_volume[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 OR price < 4h HMA (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < hma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 OR price > 4h HMA (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > hma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla breakout + 4h HMA filter
            if volume_confirmed:
                # Long entry: price > Camarilla H5 AND price > 4h HMA (bullish alignment)
                if close[i] > camarilla_h5_aligned[i] and close[i] > hma_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: price < Camarilla L5 AND price < 4h HMA (bearish alignment)
                elif close[i] < camarilla_l5_aligned[i] and close[i] < hma_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals