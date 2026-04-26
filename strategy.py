#!/usr/bin/env python3
"""
6h_RSI2_Extreme_Reversal_HTFTrendFilter_v1
Hypothesis: 6h RSI(2) extreme reversals filtered by 12h EMA50 trend. 
In strong trends (price > 12h EMA50), look for RSI(2) < 5 for long entries or RSI(2) > 95 for short entries.
Counter-trend in weak trends (price < 12h EMA50) avoided. 
Uses discrete position sizing (0.25) to minimize fee churn. 
Target: 12-37 trades/year (50-150 over 4 years) via strict RSI(2) extremes + trend alignment.
Works in bull/bear via 12h trend filter: only takes long extreme oversold in uptrend, short extreme overbought in downtrend.
"""

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
    
    # Load 12h data ONCE before loop for HTF trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for HTF trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    htf_trend = np.where(close > ema_50_12h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate RSI(2) on 6h timeframe
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = pd.Series(gain).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/2, adjust=False, min_periods=2).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_2 = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 12h EMA, 2 for RSI)
    start_idx = max(50, 2)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(rsi_2[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Extreme RSI(2) conditions with trend filter
        if htf_trend[i] == 1:  # Uptrend on 12h
            # Long extreme oversold (RSI(2) < 5)
            if rsi_2[i] < 5:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if RSI(2) > 60 (normalization)
            elif position == 1 and rsi_2[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif htf_trend[i] == -1:  # Downtrend on 12h
            # Short extreme overbought (RSI(2) > 95)
            if rsi_2[i] > 95:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if RSI(2) < 40 (normalization)
            elif position == -1 and rsi_2[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # Should not happen with our trend calculation
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RSI2_Extreme_Reversal_HTFTrendFilter_v1"
timeframe = "6h"
leverage = 1.0