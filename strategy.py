#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA trend filter and volume spike.
# Williams %R measures overbought/oversold levels: > -20 = overbought, < -80 = oversold.
# Strategy: In uptrend (price > 1w EMA50), buy when %R crosses above -80 from below (oversold bounce).
# In downtrend (price < 1w EMA50), sell when %R crosses below -20 from above (overbought rejection).
# Volume spike confirms institutional participation in both setups.
# Designed for ~15-25 trades/year per symbol (60-100 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    hh_ll = np.where(hh_ll == 0, 1e-10, hh_ll)
    
    willr = -100 * (highest_high - close) / hh_ll  # -100 to 0
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 50-period EMA on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Uptrend: price > 1w EMA50
        if close[i] > ema50_1w_aligned[i] and volume_filter[i]:
            # Buy when Williams %R crosses above -80 from below (oversold bounce)
            if willr[i] > -80 and willr[i-1] <= -80:
                signals[i] = 0.25
                position = 1
            # Exit long when %R reaches overbought (> -20) or reverse signal
            elif position == 1 and (willr[i] >= -20 or 
                                  (willr[i] < -80 and willr[i-1] >= -80)):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else 0.0
        
        # Downtrend: price < 1w EMA50
        elif close[i] < ema50_1w_aligned[i] and volume_filter[i]:
            # Sell when Williams %R crosses below -20 from above (overbought rejection)
            if willr[i] < -20 and willr[i-1] >= -20:
                signals[i] = -0.25
                position = -1
            # Exit short when %R reaches oversold (< -80) or reverse signal
            elif position == -1 and (willr[i] <= -80 or 
                                   (willr[i] > -20 and willr[i-1] <= -20)):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = -0.25 if position == -1 else 0.0
        
        # No clear trend or no volume filter: stay flat
        else:
            signals[i] = 0.0
            position = 0
    
    return signals

name = "6h_WilliamsR_1wEMA50_VolumeFilter"
timeframe = "6h"
leverage = 1.0