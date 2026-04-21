#!/usr/bin/env python3
"""
6h_1w_FundingRateMeanReversion_Bias
Hypothesis: Funding rate mean reversion works on 6h timeframe with weekly bias filter.
In BTC/ETH, extreme funding rates (> +0.03% long, < -0.03% short) reverse within 1-3 days.
Weekly EMA filter ensures we only take reversals in direction of weekly trend.
Designed for low trade frequency (15-25/year) to avoid fee drag in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load funding rate data (assuming it's available as a column)
    # If not available, we'll simulate using price action as proxy
    # In reality, funding rate would be loaded from external data
    # For now, we'll use a proxy based on price deviation from weekly VWAP
    # This approximates funding rate extremes
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Calculate weekly VWAP as proxy for fair value
    typical_price = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    vwap_1w = (typical_price * df_1w['volume'].values).cumsum() / df_1w['volume'].values.cumsum()
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Calculate deviation from weekly VWAP (proxy for funding rate extremes)
    price_dev = (prices['close'].values - vwap_1w_aligned) / vwap_1w_aligned
    
    # Smooth the deviation to avoid noise
    price_dev_smooth = pd.Series(price_dev).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    for i in range(50, n):
        # Skip if weekly EMA not ready
        if np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dev = price_dev_smooth[i]
        
        # Extreme deviation thresholds (proxy for funding rate extremes)
        extreme_long = dev < -0.005   # Price significantly below weekly VWAP
        extreme_short = dev > 0.005   # Price significantly above weekly VWAP
        
        # Weekly trend filter
        weekly_uptrend = prices['close'].iloc[i] > ema_50_1w_aligned[i]
        weekly_downtrend = prices['close'].iloc[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: extreme negative deviation + weekly uptrend bias
            if extreme_long and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: extreme positive deviation + weekly downtrend bias
            elif extreme_short and weekly_downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: deviation returns to neutral or trend breaks
            if dev > -0.001 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: deviation returns to neutral or trend breaks
            if dev < 0.001 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_FundingRateMeanReversion_Bias"
timeframe = "6h"
leverage = 1.0