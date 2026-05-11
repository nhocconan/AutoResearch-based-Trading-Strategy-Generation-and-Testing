# 1h_ICHIMOKU_TENKAN_KIJUN_CROSSOVER_4HTREND_1dVOLUMEFILTER
# Hypothesis: Ichimoku Tenkan/Kijun crossover provides reliable trend signals in both bull and bear markets.
# Use 4h Ichimoku for trend direction and 1d volume for confirmation, with 1h only for entry timing.
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year.

#!/usr/bin/env python3
name = "1h_ICHIMOKU_TENKAN_KIJUN_CROSSOVER_4HTREND_1dVOLUMEFILTER"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Ichimoku for trend direction
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    period9_high = pd.Series(high_4h).rolling(window=9, min_periods=9).max().values
    period9_low = pd.Series(low_4h).rolling(window=9, min_periods=9).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    period26_high = pd.Series(high_4h).rolling(window=26, min_periods=26).max().values
    period26_low = pd.Series(low_4h).rolling(window=26, min_periods=26).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Align to 1h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_4h, tenkan_sen)
    kijun_aligned = align_htf_to_ltf(prices, df_4h, kijun_sen)
    
    # 1d volume filter for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume_1d > vol_ma_1d * 1.5
    volume_ok_aligned = align_htf_to_ltf(prices, df_1d, volume_ok)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # Position sizing
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or 
            np.isnan(volume_ok_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Conditions
        bullish_cross = tenkan_aligned[i] > kijun_aligned[i]
        bearish_cross = tenkan_aligned[i] < kijun_aligned[i]
        
        if position == 0:
            # Long: Tenkan crosses above Kijun + 4h trend up + volume confirmation + session
            if bullish_cross and close[i] > kijun_aligned[i] and volume_ok_aligned[i] and session_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Tenkan crosses below Kijun + 4h trend down + volume confirmation + session
            elif bearish_cross and close[i] < kijun_aligned[i] and volume_ok_aligned[i] and session_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Tenkan/Kijun cross reverses OR close crosses opposite line
            if position == 1:
                if bearish_cross or close[i] < tenkan_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if bullish_cross or close[i] > tenkan_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals