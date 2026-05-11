#!/usr/bin/env python3
"""
4h_1w_Camarilla_Pivot_Sweep_TrendReversal
Hypothesis: Price sweeps through weekly Camarilla R4/S4 levels with rejection (close back inside) during low volatility, then reverses with weekly trend alignment. Works in ranging markets (reversion) and trending markets (breakout after sweep). Uses weekly structure for low-frequency, high-conviction trades.
"""

name = "4h_1w_Camarilla_Pivot_Sweep_TrendReversal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot points using previous week's OHLC
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Shift to use previous week's data (avoid look-ahead)
    w_high_prev = np.roll(w_high, 1)
    w_low_prev = np.roll(w_low, 1)
    w_close_prev = np.roll(w_close, 1)
    # First week: use current values to avoid NaN
    w_high_prev[0] = w_high[0]
    w_low_prev[0] = w_low[0]
    w_close_prev[0] = w_close[0]
    
    # Calculate pivot point
    pivot = (w_high_prev + w_low_prev + w_close_prev) / 3.0
    # Calculate Camarilla R4 and S4 levels (outer bands)
    r4 = w_close_prev + (1.1/2) * (w_high_prev - w_low_prev)
    s4 = w_close_prev - (1.1/2) * (w_high_prev - w_low_prev)
    
    # Align weekly R4/S4 to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    
    # Weekly trend filter (EMA 34)
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volatility filter: ATR(20) < ATR(50) = low volatility environment
    # Calculate ATR components
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    low_vol = atr_20 < atr_50
    low_vol = np.nan_to_num(low_vol, nan=False)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Sweep detection: price briefly breaks level but closes back inside
        swept_above = high[i] > r4_aligned[i] and close[i] < r4_aligned[i]
        swept_below = low[i] < s4_aligned[i] and close[i] > s4_aligned[i]
        
        if position == 0:
            # Long: swept below S4 (bear trap) + weekly uptrend + low vol
            if swept_below and close[i] > ema_34_1w_aligned[i] and low_vol[i]:
                signals[i] = 0.25
                position = 1
            # Short: swept above R4 (bull trap) + weekly downtrend + low vol
            elif swept_above and close[i] < ema_34_1w_aligned[i] and low_vol[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to weekly pivot or trend reversal
            if position == 1:
                # Exit long: price returns to weekly pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_34_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals