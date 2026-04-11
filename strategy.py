#!/usr/bin/env python3
"""
6h_1d_obv_divergence_v1
Strategy: 6s On-Balance Volume (OBV) divergence with 1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Uses OBV divergence to detect weakening momentum and potential reversals. 
- Bullish divergence: price makes lower low, OBV makes higher low → long
- Bearish divergence: price makes higher high, OBV makes lower high → short
Filtered by 1d EMA50 trend (only trade in direction of higher timeframe trend).
OBV is a volume-based indicator that often leads price, making it effective for early reversal detection.
In bear markets (2025+), bullish divergences at support can capture bounces; in bull markets, bearish divergences at resistance can capture pullbacks.
Designed for low trade frequency (10-30/year) with high win rate by requiring both divergence and trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_obv_divergence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate OBV
    obv = np.zeros(n)
    obv[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Lookback period for divergence detection
    lookback = 10  # 10 periods (~60 hours) to find swing points
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Find recent swing low and high in price and OBV
        # Look for lowest low and highest high in the lookback window
        price_low_idx = np.argmin(low[i-lookback:i+1]) + (i - lookback)
        price_high_idx = np.argmax(high[i-lookback:i+1]) + (i - lookback)
        obv_low_idx = np.argmin(obv[i-lookback:i+1]) + (i - lookback)
        obv_high_idx = np.argmax(obv[i-lookback:i+1]) + (i - lookback)
        
        price_low = low[price_low_idx]
        price_high = high[price_high_idx]
        obv_low = obv[obv_low_idx]
        obv_high = obv[obv_high_idx]
        
        # Current values
        price_close = close[i]
        obv_current = obv[i]
        
        # Bullish divergence: price makes lower low, OBV makes higher low
        bullish_div = (price_low < low[i-1]) and (obv_low > obv[i-1])
        # Bearish divergence: price makes higher high, OBV makes lower high
        bearish_div = (price_high > high[i-1]) and (obv_high < obv[i-1])
        
        # Trend filters
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_signal = bullish_div and uptrend_1d
        short_signal = bearish_div and downtrend_1d
        
        # Exit when opposite divergence occurs or trend changes
        exit_long = position == 1 and (bearish_div or not uptrend_1d)
        exit_short = position == -1 and (bullish_div or not downtrend_1d)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals