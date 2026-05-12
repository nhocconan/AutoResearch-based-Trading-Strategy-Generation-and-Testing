#!/usr/bin/env python3
name = "1d_PriceAction_1wTrend_Confirmation"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1-day price action: higher high/low structure for trend
    # We'll use a simple 3-bar higher high/low for uptrend, lower high/low for downtrend
    higher_high = (high > np.roll(high, 1)) & (np.roll(high, 1) > np.roll(high, 2))
    higher_low = (low > np.roll(low, 1)) & (np.roll(low, 1) > np.roll(low, 2))
    lower_high = (high < np.roll(high, 1)) & (np.roll(high, 1) < np.roll(high, 2))
    lower_low = (low < np.roll(low, 1)) & (np.roll(low, 1) < np.roll(low, 2))
    
    # Smooth the signals to avoid noise
    uptrend_raw = higher_high & higher_low
    downtrend_raw = lower_high & lower_low
    
    # Use rolling sum to require at least 2 out of 3 bars
    uptrend = pd.Series(uptrend_raw).rolling(window=3, min_periods=1).sum() >= 2
    downtrend = pd.Series(downtrend_raw).rolling(window=3, min_periods=1).sum() >= 2
    uptrend = uptrend.values
    downtrend = downtrend.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if 1w trend data not ready
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d uptrend structure + price above 1w EMA50
            if uptrend[i] and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend structure + price below 1w EMA50
            elif downtrend[i] and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when 1d trend turns down or price crosses below 1w EMA50
            if downtrend[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when 1d trend turns up or price crosses above 1w EMA50
            if uptrend[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals