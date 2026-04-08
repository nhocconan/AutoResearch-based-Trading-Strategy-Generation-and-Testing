#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Williams %R Pullback with 12h EMA Trend Filter
# Hypothesis: Williams %R pullbacks in direction of 12h EMA(50) trend capture mean reversion within trends.
# Uses 12h EMA for trend filter (works in bull/bear) and Williams %R for precise entries.
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.

name = "4h_williamsr_pullback_12h_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align 12h EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Williams %R(14) on 4h high/low/close
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R reaches overbought or trend changes
            if williams_r[i] >= -20 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: Williams %R reaches oversold or trend changes
            if williams_r[i] <= -80 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Williams %R pullback in direction of 12h trend
            if close[i] > ema_50_aligned[i]:  # Uptrend
                if williams_r[i] <= -80:  # Pullback to buy (oversold)
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                if williams_r[i] >= -20:  # Pullback to sell (overbought)
                    position = -1
                    signals[i] = -0.25
    
    return signals