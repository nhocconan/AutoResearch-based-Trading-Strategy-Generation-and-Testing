#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_weekly_pivot_bounce_v1
# Uses weekly pivot levels from 1-week chart as dynamic support/resistance.
# In bull markets: buy bounces off weekly support (S1, S2) with 6h momentum confirmation.
# In bear markets: sell bounces off weekly resistance (R1, R2) with 6m momentum confirmation.
# Uses 6h RSI(2) for short-term mean reversion entries and weekly trend filter (price vs weekly pivot).
# Target: 20-30 trades/year per symbol with high win rate via institutional levels.
name = "6h_1d_weekly_pivot_bounce_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from previous week
    high_prev = df_1w['high'].shift(1).values
    low_prev = df_1w['low'].shift(1).values
    close_prev = df_1w['close'].shift(1).values
    
    # Standard pivot point calculation
    pivot = (high_prev + low_prev + close_prev) / 3
    r1 = 2 * pivot - low_prev
    s1 = 2 * pivot - high_prev
    r2 = pivot + (high_prev - low_prev)
    s2 = pivot - (high_prev - low_prev)
    
    # Align to 6h timeframe (already delayed by 1 week due to shift)
    r1_level = align_htf_to_ltf(prices, df_1w, r1)
    r2_level = align_htf_to_ltf(prices, df_1w, r2)
    s1_level = align_htf_to_ltf(prices, df_1w, s1)
    s2_level = align_htf_to_ltf(prices, df_1w, s2)
    pivot_level = align_htf_to_ltf(prices, df_1w, pivot)
    
    # 6h RSI(2) for short-term mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use Wilder's smoothing (equivalent to RMA)
    alpha = 1.0 / 2
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # start after RSI warmup
        # Skip if levels not ready
        if np.isnan(r1_level[i]) or np.isnan(s1_level[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend: price above/below weekly pivot
        weekly_uptrend = close[i] > pivot_level[i]
        
        # Long setup: price near weekly support in uptrend with RSI oversold
        near_support = (close[i] <= s1_level[i] * 1.005) or (close[i] <= s2_level[i] * 1.005)
        rsi_oversold = rsi[i] < 20
        
        if weekly_uptrend and near_support and rsi_oversold and position != 1:
            position = 1
            signals[i] = 0.25
        
        # Short setup: price near weekly resistance in downtrend with RSI overbought
        elif not weekly_uptrend:
            near_resistance = (close[i] >= r1_level[i] * 0.995) or (close[i] >= r2_level[i] * 0.995)
            rsi_overbought = rsi[i] > 80
            
            if near_resistance and rsi_overbought and position != -1:
                position = -1
                signals[i] = -0.25
        
        # Exit conditions: RSI returns to neutral or opposite signal
        elif position == 1 and (rsi[i] > 60 or close[i] >= pivot_level[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] < 40 or close[i] <= pivot_level[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals