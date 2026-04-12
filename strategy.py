#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1w_ema_crossover_v1
# Uses 12h EMA crossover with 1-week trend filter to capture medium-term trends.
# EMA50 crossing above EMA200 = bullish, below = bearish.
# Weekly trend filter (price > weekly EMA50 for longs, < for shorts) ensures we trade with higher timeframe momentum.
# Low trade frequency expected due to EMA crossover rarity on 12h timeframe (est. 15-25 trades/year).
# Works in bull markets via trend-following crossovers and in bear markets via inverse signals.
# Includes volatility-based position sizing using ATR to normalize risk across market regimes.

name = "12h_1w_ema_crossover_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h EMAs for crossover signal
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate ATR for volatility-based position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any indicator not ready
        if np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # EMA crossover signals
        bullish_crossover = ema_50[i] > ema_200[i] and ema_50[i-1] <= ema_200[i-1]
        bearish_crossover = ema_50[i] < ema_200[i] and ema_50[i-1] >= ema_200[i-1]
        
        # Weekly trend filter: only take longs when price above weekly EMA50, shorts when below
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volatility-normalized position size (target 2.5% volatility risk)
        vol_scaled_size = 0.25 * (0.015 / atr[i])  # Scale to ~1.5% ATR risk
        vol_scaled_size = np.clip(vol_scaled_size, 0.1, 0.35)  # Keep within reasonable bounds
        
        # Entry logic: crossover in direction of weekly trend
        if bullish_crossover and weekly_uptrend and position != 1:
            position = 1
            signals[i] = vol_scaled_size
        elif bearish_crossover and weekly_downtrend and position != -1:
            position = -1
            signals[i] = -vol_scaled_size
        
        # Exit logic: opposite crossover
        elif bearish_crossover and position == 1:
            position = 0
            signals[i] = 0.0
        elif bullish_crossover and position == -1:
            position = 0
            signals[i] = 0.0
        
        # Hold position
        else:
            if position == 1:
                signals[i] = vol_scaled_size
            elif position == -1:
                signals[i] = -vol_scaled_size
            else:
                signals[i] = 0.0
    
    return signals