#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price crossing above/below 1-day VWAP with volume confirmation and 1-week trend filter.
# Long when price crosses above 1-day VWAP, volume > 1.5x 20-period average, and price > 1-week EMA50.
# Short when price crosses below 1-day VWAP, volume > 1.5x 20-period average, and price < 1-week EMA50.
# Exit when price crosses back below/above 1-day VWAP.
# Uses VWAP for intraday mean reversion with higher timeframe trend alignment to avoid counter-trend trades.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "6h_VWAP_1dVWAP_1wEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 6h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # 6h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1-day VWAP (using daily typical price and volume)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_numerator_1d = np.cumsum(typical_price_1d * df_1d['volume'].values)
    vwap_denominator_1d = np.cumsum(df_1d['volume'].values)
    vwap_1d = vwap_numerator_1d / vwap_denominator_1d
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 1-week EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for VWAP and EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price crosses above 1d VWAP, volume filter, above 1w EMA50
            long_cond = (close[i] > vwap_1d_aligned[i]) and (close[i-1] <= vwap_1d_aligned[i-1]) and volume_filter[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: price crosses below 1d VWAP, volume filter, below 1w EMA50
            short_cond = (close[i] < vwap_1d_aligned[i]) and (close[i-1] >= vwap_1d_aligned[i-1]) and volume_filter[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below 1d VWAP
            if close[i] < vwap_1d_aligned[i] and close[i-1] >= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above 1d VWAP
            if close[i] > vwap_1d_aligned[i] and close[i-1] <= vwap_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals