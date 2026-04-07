#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R Reversal with 1d Trend Filter
# Hypothesis: Williams %R overbought/oversold reversals in direction of 1d EMA(50) trend
# capture mean reversion within trends. Works in both bull and bear markets by
# filtering trades with the higher timeframe trend. Target: 15-35 trades/year.

name = "6h_williamsr_reversal_1d_ema_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d EMA to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R(14) on 6h
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - lookback + 1)
        highest_high[i] = np.max(high[start_idx:i+1])
        lowest_low[i] = np.min(low[start_idx:i+1])
    
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close) / (highest_high - lowest_low),
        -50  # neutral when range is zero
    )
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R reaches oversold or trend changes
            if williams_r[i] <= -80 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Williams %R reaches overbought or trend changes
            if williams_r[i] >= -20 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Williams %R reversal in direction of 1d trend
            if close[i] > ema_50_aligned[i]:  # Uptrend
                if williams_r[i] >= -80 and williams_r[i] <= -50:  # Oversold reversal
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                if williams_r[i] <= -20 and williams_r[i] >= -50:  # Overbought reversal
                    position = -1
                    signals[i] = -0.25
    
    return signals