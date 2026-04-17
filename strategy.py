#!/usr/bin/env python3
"""
Hypothesis: On the 6-hour timeframe, price tends to revert to the 1-week Volume Weighted Average Price (VWAP) after significant deviations.
We use 1-week VWAP as a dynamic mean reversion target, with entry triggered when price deviates beyond 2 standard deviations
and shows signs of reversal (price closing back toward VWAP). Trend filter uses 1-day EMA50 to avoid counter-trend trades.
Designed for 6h to work in both trending and ranging markets by fading extremes in sideways regimes and filtering with trend.
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
    
    # Get 1d and 1w data for indicators
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week VWAP (typical price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_num = (typical_price * df_1w['volume']).cumsum()
    vwap_den = df_1w['volume'].cumsum()
    vwap = vwap_num / vwap_den
    
    # Calculate standard deviation of price from VWAP over past 20 periods
    price_dev = typical_price - vwap
    vwap_std = price_dev.rolling(window=20, min_periods=20).std()
    
    # Upper and lower bands: VWAP ± 2 * std
    vwap_upper = vwap + 2.0 * vwap_std
    vwap_lower = vwap - 2.0 * vwap_std
    
    # Calculate 1-day EMA50 for trend filter
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1w indicators to 6h timeframe
    vwap_6h = align_htf_to_ltf(prices, df_1w, vwap.values)
    vwap_upper_6h = align_htf_to_ltf(prices, df_1w, vwap_upper.values)
    vwap_lower_6h = align_htf_to_ltf(prices, df_1w, vwap_lower.values)
    
    # Align 1d EMA50
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for 1w VWAP std and 1d EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_6h[i]) or np.isnan(vwap_upper_6h[i]) or np.isnan(vwap_lower_6h[i]) or
            np.isnan(ema_50_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price below lower band and closing back toward VWAP (reversal signal)
            if price < vwap_lower_6h[i] and price > close[i-1] and price > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price above upper band and closing back toward VWAP (reversal signal)
            elif price > vwap_upper_6h[i] and price < close[i-1] and price < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or breaks below lower band (failed mean reversion)
            if price >= vwap_6h[i] or price < vwap_lower_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or breaks above upper band (failed mean reversion)
            if price <= vwap_6h[i] or price > vwap_upper_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wVWAP_2StdDev_Reversal"
timeframe = "6h"
leverage = 1.0