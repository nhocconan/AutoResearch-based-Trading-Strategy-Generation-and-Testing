#!/usr/bin/env python3
# Hypothesis: 4h price near weekly pivot levels with daily trend filter and volume confirmation
# Long when price above weekly pivot, above daily EMA50, and volume > 1.5x 20-period average
# Short when price below weekly pivot, below daily EMA50, and volume > 1.5x 20-period average
# Exit when price crosses back below/above weekly pivot OR EMA direction contradicts position
# Uses weekly pivot for structure and daily EMA for trend - balances signal frequency for 4h timeframe
# Position size: 0.25 (25% of capital) to manage drawdown while maintaining profitability

name = "4h_WeeklyPivot_DailyEMA_Volume"
timeframe = "4h"
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
    volume = prices['volume'].values
    
    # Daily EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get weekly data for pivot points (weekly high, low, close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Support and resistance levels
    R1 = 2 * pivot - df_1w['low']
    S1 = 2 * pivot - df_1w['high']
    
    # Align weekly pivot levels to 4h timeframe (waits for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot.values)
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1.values)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above weekly pivot AND above daily EMA50 (bullish alignment) + volume spike
            if (close[i] > pivot_aligned[i] and 
                close[i] > ema50[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly pivot AND below daily EMA50 (bearish alignment) + volume spike
            elif (close[i] < pivot_aligned[i] and 
                  close[i] < ema50[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly pivot OR EMA50 turns bearish
            if (close[i] < pivot_aligned[i]) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly pivot OR EMA50 turns bullish
            if (close[i] > pivot_aligned[i]) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals