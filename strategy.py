# 4H_CAMARILLA_R1S1_BREAKOUT_1W_TREND_VOLUME_CONFIRMED
# Hypothesis: Price breaking above R1 or below S1 with weekly trend alignment and volume confirmation
# captures institutional breakout moves while avoiding false signals in chop. Weekly trend filter
# reduces whipsaws in sideways markets. Volume > 1.5x average confirms institutional participation.
# Target: 80-120 total trades over 4 years (20-30/year) with size 0.25 to balance opportunity and cost.
name = "4H_CAMARILLA_R1S1_BREAKOUT_1W_TREND_VOLUME_CONFIRMED"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate weekly OHLC for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least 2 weeks for previous week data
        return np.zeros(n)
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = df_1w['high'].shift(1)
    prev_low = df_1w['low'].shift(1)
    prev_close = df_1w['close'].shift(1)
    
    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    # Calculate Camarilla levels (R1, S1 are primary levels)
    r1 = pp + (prev_high - prev_low) * 1.0833
    s1 = pp - (prev_high - prev_low) * 1.0833
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1, weekly EMA50 uptrend, volume confirmation
            if (close[i] > r1_aligned[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, weekly EMA50 downtrend, volume confirmation
            elif (close[i] < s1_aligned[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot point or breaks below S1
            if (close[i] <= pp_aligned[i]) or (close[i] < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot point or breaks above R1
            if (close[i] >= pp_aligned[i]) or (close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals