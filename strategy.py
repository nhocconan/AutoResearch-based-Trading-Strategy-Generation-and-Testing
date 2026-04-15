#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with Elder Ray and Volume Confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) to identify trend direction.
# Elder Ray (Bull/Bear Power) confirms momentum strength.
# Volume filter ensures participation. Works in both bull and bear markets by
# following the Alligator's alignment. Target: 50-150 total trades.
# Timeframe: 12h, HTF: 1w/1d for trend context.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator (13,8,5) - Smoothed Median Prices
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().rolling(8).mean()
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().rolling(5).mean()
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().rolling(3).mean()
    
    jaw = jaw.values
    teeth = teeth.values
    lips = lips.values
    
    # Elder Ray Power (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 1d data for trend context (higher timeframe filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_1d_aligned[i])):
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = Uptrend
        # Lips < Teeth < Jaw = Downtrend
        is_uptrend = lips[i] > teeth[i] > jaw[i]
        is_downtrend = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        bull_rising = (i > 0 and bull_power[i] > bull_power[i-1])
        bear_falling = (i > 0 and bear_power[i] < bear_power[i-1])
        
        # Volume confirmation: above average volume
        vol_ma = np.mean(volume[max(0, i-10):i+1])
        vol_surge = volume[i] > 1.5 * vol_ma
        
        # Long entry: Uptrend + Bull Power rising + volume surge + price above 1d EMA50
        if (is_uptrend and bull_rising and vol_surge and close[i] > ema50_1d_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Downtrend + Bear Power falling + volume surge + price below 1d EMA50
        elif (is_downtrend and bear_falling and vol_surge and close[i] < ema50_1d_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Alligator reverses or Elder Ray diverges
        elif position == 1 and (not is_uptrend or not bull_rising):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not is_downtrend or not bear_falling):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Williams_Alligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0