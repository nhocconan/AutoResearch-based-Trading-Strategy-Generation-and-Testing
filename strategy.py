#!/usr/bin/env python3
name = "6h_Liquidity_Sweep_Reversal_Volume"
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
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d daily range for liquidity levels (previous day high/low)
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    
    # Align 1d liquidity levels to 6h timeframe
    liq_high_1d = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    liq_low_1d = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # 1w trend filter: price above/below 20-period EMA
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # 6h volume filter: > 2x 24-period average (6h * 24 = 6 days)
    vol_ma_6h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 2.0 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Wait for volume MA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(liq_high_1d[i]) or np.isnan(liq_low_1d[i]) or 
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: sweep below previous day low then reverse with volume
            # Look for rejection of lows: current low touches/slightly breaks liq_low then closes back above
            if (low[i] <= liq_low_1d[i] * 1.002 and  # Allow 0.2% slippage for sweep
                close[i] > liq_low_1d[i] and        # Close back above liquidity low
                vol_filter[i] and                   # Volume confirmation
                close[i] > ema20_1w_aligned[i]):    # Weekly uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: sweep above previous day high then reverse with volume
            elif (high[i] >= liq_high_1d[i] * 0.998 and  # Allow 0.2% slippage for sweep
                  close[i] < liq_high_1d[i] and          # Close back below liquidity high
                  vol_filter[i] and                      # Volume confirmation
                  close[i] < ema20_1w_aligned[i]):       # Weekly downtrend filter
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below liquidity low or weekly trend fails
            if close[i] < liq_low_1d[i] * 0.998 or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above liquidity high or weekly trend fails
            if close[i] > liq_high_1d[i] * 1.002 or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Liquidity sweeps (stop hunts) at daily high/low levels create reversal opportunities
# with institutional order flow. Volume confirms genuine reversal vs fakeout. Weekly EMA filter
# ensures alignment with higher timeframe trend. Works in both bull/bear markets as liquidity
# sweeps occur in all conditions. Target: 15-25 trades/year per symbol to minimize fee drag.