#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation
# The Williams Alligator (13,8,5 SMAs) identifies trends when jaws, teeth, and lips are aligned.
# Long when lips > teeth > jaws (bullish alignment) and 12h EMA(34) uptrend and volume spike.
# Short when lips < teeth < jaws (bearish alignment) and 12h EMA(34) downtrend and volume spike.
# Volume spike confirms institutional participation; avoids choppy false signals.
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost.

name = "6h_WilliamsAlligator_12hTrend_Volume"
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
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    daily_close = df_12h['close'].values
    ema34_12h = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Williams Alligator: SMAs of median price
    # Jaws: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    median_price = (high + low) / 2.0
    
    # Calculate SMAs using rolling mean (simple moving average)
    jaws_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Shift to align with Alligator logic
    jaws = np.roll(jaws_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set invalid values for shifted periods
    jaws[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(jaws[i]) or 
            np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_12h_val = ema34_12h_aligned[i]
        price = close[i]
        jaw = jaws[i]
        tooth = teeth[i]
        lip = lips[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: bullish alignment (lip > tooth > jaw) and 12h uptrend and volume spike
            if lip > tooth and tooth > jaw and close > ema34_12h_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: bearish alignment (lip < tooth < jaw) and 12h downtrend and volume spike
            elif lip < tooth and tooth < jaw and close < ema34_12h_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment breaks or 12h trend turns down
            if lip <= tooth or tooth <= jaw or close < ema34_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment breaks or 12h trend turns up
            if lip >= tooth or tooth >= jaw or close > ema34_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals