# 6h_Pivot_R1_S1_Breakout_Volume_ATRFilter
# Hypothesis: Pivot point breakouts with volume confirmation and ATR volatility filter
# Perform well in both trending and ranging markets by filtering false breakouts.
# Uses daily pivot levels (R1, S1) on 6h chart with volume > 1.5x average and ATR > 0.5*ATRmean.
# Designed for 60-120 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(low, 1))
    tr3 = np.abs(low - np.roll(high, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily pivot points from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pp = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # warmup for volume MA
        # Skip if any data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 50% of its 50-period average
        atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean()
        atr_ma50_val = atr_ma50.iloc[i] if not np.isnan(atr_ma50.iloc[i]) else atr[i]
        vol_filter = atr[i] > 0.5 * atr_ma50_val
        
        # Volume filter
        vol_filter = vol_filter and (volume[i] > 1.5 * volume_ma20[i])
        
        # Breakout conditions
        long_breakout = close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]
        short_breakout = close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]
        
        # Entry logic
        if position == 0:
            if long_breakout and vol_filter:
                signals[i] = 0.25
                position = 1
            elif short_breakout and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or volatility drops
            if close[i] < s1_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or volatility drops
            if close[i] > r1_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_ATRFilter"
timeframe = "6h"
leverage = 1.0