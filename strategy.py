#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 123 Reversal Pattern + 1d Volume Spike + Trend Filter
# The 123 Reversal is a price action pattern where:
# - Point 1: Recent swing high/low
# - Point 2: Pullback in opposite direction
# - Point 3: Failed retest of Point 1, signaling reversal
# We identify Point 3 as a failed breakout of the recent swing point with
# confirmation from 1d volume spike (institutional interest) and 1d EMA55 trend filter.
# This captures institutional reversal attempts with lower false signals.
# Target: 20-40 trades/year (~80-160 total) for low frequency and high accuracy.

name = "6h_123Reversal_1dVolume_EMA55"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Identify swing points: lookback 5 periods for local high/low
    def find_swing_points(arr, lookback=5):
        swings_high = np.full_like(arr, np.nan)
        swings_low = np.full_like(arr, np.nan)
        for i in range(lookback, len(arr)-lookback):
            if arr[i] == np.max(arr[i-lookback:i+lookback+1]):
                swings_high[i] = arr[i]
            if arr[i] == np.min(arr[i-lookback:i+lookback+1]):
                swings_low[i] = arr[i]
        return swings_high, swings_low
    
    swing_high, swing_low = find_swing_points(high, 5)
    
    # Get 1d data for volume spike and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA55 for trend filter
    close_1d = df_1d['close'].values
    ema55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    ema55_1d_slope = ema55_1d[1:] - ema55_1d[:-1]
    ema55_1d_slope = np.concatenate([[0], ema55_1d_slope])
    ema55_1d_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)
    ema55_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d_slope)
    
    # 1d volume spike: current volume > 2.0x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for swing points and indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(swing_high[i]) or np.isnan(swing_low[i]) or 
            np.isnan(ema55_1d_aligned[i]) or np.isnan(ema55_1d_slope_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price action
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        
        # Recent swing points (look back 1-10 bars for validity)
        lookback = min(10, i-5)
        if lookback < 5:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Find most recent swing high and low within lookback
        recent_swing_high = np.nan
        recent_swing_low = np.nan
        for j in range(i-lookback, i+1):
            if not np.isnan(swing_high[j]):
                recent_swing_high = swing_high[j]
            if not np.isnan(swing_low[j]):
                recent_swing_low = swing_low[j]
        
        if position == 0:
            # Bearish 123 reversal (potential short):
            # 1. Point 1: recent swing high
            # 2. Point 2: pullback low (after Point 1)
            # 3. Point 3: failed retest of Point 1 (fails to break above)
            if not np.isnan(recent_swing_high):
                # Find pullback low after the swing high
                pullback_low = np.inf
                high_formed = False
                for j in range(i-lookback, i+1):
                    if high[j] >= recent_swing_high * 0.999:  # near swing high
                        high_formed = True
                    if high_formed and low[j] < pullback_low:
                        pullback_low = low[j]
                
                # Point 3: current high fails to break above swing high
                if (curr_high < recent_swing_high and 
                    pullback_low < recent_swing_high and  # valid pullback
                    ema55_1d_slope_aligned[i] < 0 and    # 1d downtrend
                    vol_spike_1d_aligned[i]):            # volume confirmation
                    signals[i] = -0.25
                    position = -1
            
            # Bullish 123 reversal (potential long):
            # 1. Point 1: recent swing low
            # 2. Point 2: pullback high (after Point 1)
            # 3. Point 3: failed retest of Point 1 (fails to break below)
            if not np.isnan(recent_swing_low):
                # Find pullback high after the swing low
                pullback_high = -np.inf
                low_formed = False
                for j in range(i-lookback, i+1):
                    if low[j] <= recent_swing_low * 1.001:  # near swing low
                        low_formed = True
                    if low_formed and high[j] > pullback_high:
                        pullback_high = high[j]
                
                # Point 3: current low fails to break below swing low
                if (curr_low > recent_swing_low and 
                    pullback_high > recent_swing_low and   # valid pullback
                    ema55_1d_slope_aligned[i] > 0 and     # 1d uptrend
                    vol_spike_1d_aligned[i]):             # volume confirmation
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Exit long: price breaks above recent swing high (stop loss) or trend/volume fails
            if (curr_high > recent_swing_high or 
                ema55_1d_slope_aligned[i] <= 0 or 
                not vol_spike_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks below recent swing low (stop loss) or trend/volume fails
            if (curr_low < recent_swing_low or 
                ema55_1d_slope_aligned[i] >= 0 or 
                not vol_spike_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals