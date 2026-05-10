#!/usr/bin/env python3
# 1d_Williams_Alligator_Elder_Ray_Momentum
# Hypothesis: Williams Alligator (SMAs 13/8/5) combined with Elder Ray (Bull/Bear Power) provides clear trend direction.
# Long when price above Alligator teeth (13-period SMA) and Bull Power > 0.
# Short when price below Alligator teeth and Bear Power < 0.
# Uses 1-week trend filter to align with higher timeframe momentum.
# Designed for low-frequency trading (target: 10-25 trades/year) to minimize fee impact in ranging markets.

name = "1d_Williams_Alligator_Elder_Ray_Momentum"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Williams Alligator on daily chart
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (N-1) + CLOSE) / N
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)   # Jaw (blue line)
    teeth = smma(close, 8)  # Teeth (red line)
    lips = smma(close, 5)   # Lips (green line)
    
    # Calculate Elder Ray indicators
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator (need 13 for jaw, 13 for EMA)
    start_idx = max(13, 13)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Alligator alignment: check if jaws, teeth, lips are properly aligned
        # For uptrend: Lips > Teeth > Jaw (green > red > blue)
        # For downtrend: Jaw > Teeth > Lips (blue > red > green)
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray confirmation
        bull_confirm = bull_power[i] > 0
        bear_confirm = bear_power[i] < 0
        
        if position == 0:
            # Long entry: price above teeth + weekly uptrend + Alligator aligned up + Bull Power positive
            if close[i] > teeth[i] and weekly_uptrend and alligator_long and bull_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price below teeth + weekly downtrend + Alligator aligned down + Bear Power negative
            elif close[i] < teeth[i] and weekly_downtrend and alligator_short and bear_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below teeth or Alligator turns down or weekly trend turns down
            if (close[i] < teeth[i] or not alligator_long or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above teeth or Alligator turns up or weekly trend turns up
            if (close[i] > teeth[i] or not alligator_short or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals