#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h Donchian Channel (20-period)
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d daily pivot points (standard: PP, R1/R2/R3, S1/S2/S3)
    # Pivot Point = (High + Low + Close) / 3
    pp_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Resistance levels
    r1_1d = 2 * pp_1d - df_1d['low']
    r2_1d = pp_1d + (df_1d['high'] - df_1d['low'])
    r3_1d = df_1d['high'] + 2 * (pp_1d - df_1d['low'])
    # Support levels
    s1_1d = 2 * pp_1d - df_1d['high']
    s2_1d = pp_1d - (df_1d['high'] - df_1d['low'])
    s3_1d = df_1d['low'] - 2 * (df_1d['high'] - pp_1d)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Align daily pivot levels to 6h timeframe (with 1-bar delay for completed daily bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_avg[i]) or
            np.isnan(pp_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5  # Volume must be 1.5x average
        
        if position == 0:
            # Long setup: Break above Donchian high AND above daily R3 (strong bullish breakout) AND volume confirmation
            if (price > donchian_high[i] and price > r3_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: Break below Donchian low AND below daily S3 (strong bearish breakout) AND volume confirmation
            elif (price < donchian_low[i] and price < s3_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price retracement to Donchian middle
            if price < donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price retracement to Donchian middle
            if price > donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Donchian_DailyPivot_Breakout_Volume"
timeframe = "6h"
leverage = 1.0