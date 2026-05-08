#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour SuperTrend with 1-week trend filter and volume confirmation
# Long when SuperTrend gives buy signal + weekly EMA(20) uptrend + volume spike
# Short when SuperTrend gives sell signal + weekly EMA(20) downtrend + volume spike
# SuperTrend adapts to volatility and captures trends effectively
# Weekly trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_SuperTrend_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # SuperTrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize SuperTrend
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    # Calculate SuperTrend
    for i in range(1, len(close)):
        if np.isnan(atr[i]) or np.isnan(hl2[i]):
            supertrend[i] = supertrend[i-1] if i > 0 else hl2[i]
            direction[i] = direction[i-1] if i > 0 else 1
            continue
            
        # Upper and lower band logic
        if upper_band[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
            upper_band[i] = upper_band[i]
        else:
            upper_band[i] = upper_band[i-1]
            
        if lower_band[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
            lower_band[i] = lower_band[i]
        else:
            lower_band[i] = lower_band[i-1]
        
        # SuperTrend logic
        if supertrend[i-1] == upper_band[i-1]:
            if close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(atr_period, 20)  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(supertrend[i]) or 
            np.isnan(direction[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        st_direction = direction[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: SuperTrend uptrend + weekly uptrend + volume spike
            if st_direction == 1 and close[i] > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: SuperTrend downtrend + weekly downtrend + volume spike
            elif st_direction == -1 and close[i] < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: SuperTrend turns down OR weekly trend turns down
            if st_direction == -1 or close[i] < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: SuperTrend turns up OR weekly trend turns up
            if st_direction == 1 or close[i] > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals