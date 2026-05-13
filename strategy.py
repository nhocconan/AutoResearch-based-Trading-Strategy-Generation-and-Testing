#!/usr/bin/env python3
# Hypothesis: 6h Volume-Weighted Average Price (VWAP) deviation with 1w trend filter and 1d volume confirmation.
# Long when price > 6h VWAP AND 1w close > 1w EMA200 (bullish regime) AND 1d volume > 1.5x 20-period average
# Short when price < 6h VWAP AND 1w close < 1w EMA200 (bearish regime) AND 1d volume > 1.5x 20-period average
# Exit when price crosses back below/above 6h VWAP OR 1w trend reverses
# Uses 6h for lower frequency, VWAP for mean reversion within trend, 1w EMA200 for regime filter, 1d volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via continuation above VWAP, bear via faded rallies below VWAP.

name = "6h_VWAP_1wTrend_1dVolume_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 6h data for VWAP calculation
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # Calculate typical price and VWAP components for 6h
    typical_price_6h = (high_6h + low_6h + close_6h) / 3.0
    vol_tp_6h = typical_price_6h * volume_6h
    cum_vol_tp_6h = np.cumsum(vol_tp_6h)
    cum_vol_6h = np.cumsum(volume_6h)
    vwap_6h = np.divide(cum_vol_tp_6h, cum_vol_6h, out=np.zeros_like(cum_vol_tp_6h), where=cum_vol_6h!=0)
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Get 1w data for trend filter (EMA200 on close)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(vwap_6h[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(volume_filter_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > 6h VWAP AND 1w close > 1w EMA200 (bullish regime) AND 1d volume confirmation
            if close[i] > vwap_6h[i] and close_1w[i] > ema200_1w_aligned[i] and volume_filter_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < 6h VWAP AND 1w close < 1w EMA200 (bearish regime) AND 1d volume confirmation
            elif close[i] < vwap_6h[i] and close_1w[i] < ema200_1w_aligned[i] and volume_filter_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below 6h VWAP OR 1w trend reverses (close < EMA200)
            if close[i] < vwap_6h[i] or close_1w[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above 6h VWAP OR 1w trend reverses (close > EMA200)
            if close[i] > vwap_6h[i] or close_1w[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals