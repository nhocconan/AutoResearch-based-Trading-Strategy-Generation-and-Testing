#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 12h EMA50 trend filter
# Uses Jaw/Teeth/Lips for trend direction, Bull/Bear Power for momentum confirmation.
# 12h EMA50 ensures we trade with higher timeframe trend. Discrete sizing 0.25.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via trend+momentum alignment,
# and in bear via short signals when alligator is bearish and Elder Ray confirms weakness.

name = "6h_WilliamsAlligator_ElderRay_12hEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs of median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 13, 8, 5)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_50_12h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish: Lips > Teeth > Jaw (aligned) + Bull Power > 0 + close above 12h EMA50
            if (curr_lips > curr_teeth > curr_jaw and 
                curr_bull_power > 0 and 
                curr_close > curr_ema_50_12h):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Bearish: Jaw > Teeth > Lips (aligned) + Bear Power < 0 + close below 12h EMA50
            elif (curr_jaw > curr_teeth > curr_lips and 
                  curr_bear_power < 0 and 
                  curr_close < curr_ema_50_12h):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Alligator reverses OR Bull Power turns negative OR loses 12h trend
            if not (curr_lips > curr_teeth > curr_jaw) or curr_bull_power <= 0 or curr_close < curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator reverses OR Bear Power turns positive OR loses 12h trend
            if not (curr_jaw > curr_teeth > curr_lips) or curr_bear_power >= 0 or curr_close > curr_ema_50_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals