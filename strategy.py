#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with weekly trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Weekly trend: EMA(26) slope positive = bull market, negative = bear market.
# In bull market: long when Bull Power > 0 and rising, exit when Bull Power < 0.
# In bear market: short when Bear Power < 0 and falling, exit when Bear Power > 0.
# Volume confirmation: volume > 1.5x 20-period average to avoid low-volume false signals.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "6h_ElderRay_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(26) for trend
    ema26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    # Slope of EMA26: positive = bull trend, negative = bear trend
    ema26_slope = np.diff(ema26_1w, prepend=ema26_1w[0])
    
    # Calculate 6h EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = low - ema13   # Bear Power = Low - EMA
    
    # Align weekly trend to 6h
    ema26_slope_aligned = align_htf_to_ltf(prices, df_1w, ema26_slope)
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20)  # Ensure EMA26(26) and volume MA(20) are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema26_slope_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bull = bull_power[i]
        bear = bear_power[i]
        trend_slope = ema26_slope_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Look for new entries
            if trend_slope > 0:  # Bull market (weekly uptrend)
                # Long when Bull Power positive and rising
                if bull > 0 and (i == start_idx or bull > bull_power[i-1]):
                    if volume_confirmed:
                        signals[i] = 0.25
                        position = 1
            else:  # Bear market (weekly downtrend)
                # Short when Bear Power negative and falling (more negative)
                if bear < 0 and (i == start_idx or bear < bear_power[i-1]):
                    if volume_confirmed:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long position: exit when Bull Power turns negative
            if bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit when Bear Power turns positive
            if bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals