#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend(ATR=10, mult=3.0) + 1w EMA(50) trend filter
# - Uses 6h Supertrend for trend direction and entry timing
# - Uses 1-week EMA(50) as higher timeframe trend filter: only trade long when price > EMA50_1w, short when price < EMA50_1w
# - Supertrend provides dynamic support/resistance with ATR-based trailing stops
# - Weekly EMA filter ensures we only trade in alignment with the major trend, reducing whipsaws in ranging markets
# - Works in both bull and bear markets by following the weekly trend
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_supertrend_ema_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(10) for Supertrend
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend components
    hl2 = (high + low) / 2
    upper_band = hl2 + 3.0 * atr
    lower_band = hl2 - 3.0 * atr
    
    # Initialize Supertrend arrays
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(10, n):
        # Upper band logic
        if i == 10:
            upper_band[i] = hl2[i] + 3.0 * atr[i]
            lower_band[i] = hl2[i] - 3.0 * atr[i]
        else:
            upper_band[i] = hl2[i] + 3.0 * atr[i]
            if upper_band[i] > upper_band[i-1] or close[i-1] <= upper_band[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            lower_band[i] = hl2[i] - 3.0 * atr[i]
            if lower_band[i] < lower_band[i-1] or close[i-1] >= lower_band[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = lower_band[i-1]
        
        # Trend direction
        if i == 10:
            direction[i] = 1 if close[i] > upper_band[i] else -1
        else:
            if direction[i-1] == -1 and close[i] > upper_band[i]:
                direction[i] = 1
            elif direction[i-1] == 1 and close[i] < lower_band[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
        
        # Supertrend value
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(supertrend[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i]) or atr[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only trade in direction of weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price closes below Supertrend (trend reversal)
            if close[i] < supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price closes above Supertrend (trend reversal)
            if close[i] > supertrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price above Supertrend AND weekly uptrend
            if close[i] > supertrend[i] and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Enter short: price below Supertrend AND weekly downtrend
            elif close[i] < supertrend[i] and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals