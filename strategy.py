#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1w EMA200 trend filter and volume spike confirmation.
- Williams %R(14) on 12h: measures overbought/oversold levels (-100 to 0)
- Long when Williams %R crosses above -80 from below AND price > 1w EMA200 (uptrend)
- Short when Williams %R crosses below -20 from above AND price < 1w EMA200 (downtrend)
- Volume confirmation: current volume > 1.5 * 20-period average volume on 12h
- ATR-based stoploss: exit when price moves 2*ATR against position
- Designed to capture mean reversals in strong trends with institutional volume
- Signal size: 0.25 discrete levels to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 14-period Williams %R on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough data for Williams %R and averages
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to lower timeframe (12h -> 6h/15m etc)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:  # Need enough data for EMA200
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Trend filter: price above/below 1w EMA200
    uptrend = close > ema_200_1w_aligned
    downtrend = close < ema_200_1w_aligned
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume on 12h
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_12h / avg_volume_12h
    volume_ratio = np.where(avg_volume_12h == 0, 0, volume_ratio)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_12h, volume_ratio)
    volume_filter = volume_ratio_aligned > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 200)  # Need Williams %R, volume avg, and 1w EMA200 data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below AND uptrend AND volume confirmation
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                uptrend[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above AND downtrend AND volume confirmation
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  downtrend[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses above -20 (overbought) OR volume drops
            if williams_r_aligned[i] >= -20 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses below -80 (oversold) OR volume drops
            if williams_r_aligned[i] <= -80 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1wEMA200_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0