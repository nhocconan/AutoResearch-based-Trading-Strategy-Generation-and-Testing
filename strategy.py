#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1-day high/low for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day pivot points (standard)
    pp = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pp - low_1d
    s1 = 2 * pp - high_1d
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = high_1d + 2 * (pp - low_1d)
    s3 = low_1d - 2 * (high_1d - pp)
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current > 1.3x median of last 20 periods
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.3 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long conditions:
        # 1. Price above daily pivot (bullish bias)
        # 2. Pullback to S1 or S2 with volume confirmation
        # 3. Volatility filter: ATR > 0
        if (close[i] > pp_aligned[i] and 
            ((close[i] <= s1_aligned[i] * 1.005) or (close[i] <= s2_aligned[i] * 1.005)) and
            volume[i] > vol_threshold[i] and atr14[i] > 0):
            signals[i] = 0.25
        
        # Short conditions:
        # 1. Price below daily pivot (bearish bias)
        # 2. Pullback to R1 or R2 with volume confirmation
        # 3. Volatility filter: ATR > 0
        elif (close[i] < pp_aligned[i] and 
              ((close[i] >= r1_aligned[i] * 0.995) or (close[i] >= r2_aligned[i] * 0.995)) and
              volume[i] > vol_threshold[i] and atr14[i] > 0):
            signals[i] = -0.25
        
        # Exit conditions: price crosses daily pivot
        elif i > 0:
            if (signals[i-1] == 0.25 and close[i] < pp_aligned[i]) or \
               (signals[i-1] == -0.25 and close[i] > pp_aligned[i]):
                signals[i] = 0.0
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_DailyPivot_Pullback_Volume"
timeframe = "6h"
leverage = 1.0