#!/usr/bin/env python3
"""
6H_Elder_Ray_Combined_1W_Trend_v1
Hypothesis: Combine Elder Ray (bull/bear power) with 1-week trend filter.
Long when bull power > 0 and bear power < 0 (bullish) and price > 1-week EMA200.
Short when bear power > 0 and bull power < 0 (bearish) and price < 1-week EMA200.
Uses 6-period EMA for Elder Ray calculation. Filters trades to only occur in strong weekly trends.
Designed to work in both bull and bear markets by following the higher timeframe trend.
Target: 50-150 total trades over 4 years (12-37/year).
"""
name = "6H_Elder_Ray_Combined_1W_Trend_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA200 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema200_1w = close_1w.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Elder Ray on 6h data
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 200)  # Ensure EMA13 and EMA200 are ready
    
    for i in range(start_idx, n):
        # Skip if 1w EMA200 not ready
        if np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish Elder Ray + price above 1w EMA200
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Elder Ray + price below 1w EMA200
            elif bear_power[i] > 0 and bull_power[i] < 0 and close[i] < ema200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Elder Ray divergence or price crosses 1w EMA200
            if position == 1:
                if bull_power[i] <= 0 or bear_power[i] >= 0 or close[i] < ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            elif position == -1:
                if bear_power[i] <= 0 or bull_power[i] >= 0 or close[i] > ema200_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals