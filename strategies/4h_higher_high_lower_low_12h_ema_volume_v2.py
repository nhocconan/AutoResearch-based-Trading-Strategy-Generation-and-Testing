#!/usr/bin/env python3
# 4h_higher_high_lower_low_12h_ema_volume_v2
# Hypothesis: 4h price action forming higher highs/lows (HH/LL) in uptrend or lower highs/lows (LH/LL) in downtrend,
# confirmed by 12h EMA trend and volume spike. Exits on trend reversal or mean reversion to 12h EMA.
# Works in bull/bear: 12h EMA filters trend direction, HH/LL structure captures momentum, volume ensures validity.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_higher_high_lower_low_12h_ema_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA(21)
    ema_12h = pd.Series(close_12h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR price reverts to 12h EMA (mean reversion)
            if close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR price reverts to 12h EMA (mean reversion)
            if close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need at least 2 bars to check HH/LL structure
            if i < 2:
                signals[i] = 0.0
                continue
                
            # Volume confirmation
            volume_confirmed = volume[i] > 1.8 * volume_ma[i]
            
            if volume_confirmed:
                # Check for HH/LL structure (uptrend)
                hh_ll = (high[i] > high[i-1]) and (low[i] > low[i-1])
                # Check for LH/LL structure (downtrend)
                lh_ll = (high[i] < high[i-1]) and (low[i] < low[i-1])
                
                if hh_ll and close[i] > ema_12h_aligned[i]:
                    # Uptrend structure + price above 12h EMA → long
                    position = 1
                    signals[i] = 0.25
                elif lh_ll and close[i] < ema_12h_aligned[i]:
                    # Downtrend structure + price below 12h EMA → short
                    position = -1
                    signals[i] = -0.25
    
    return signals