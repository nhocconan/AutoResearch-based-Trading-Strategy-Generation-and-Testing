#!/usr/bin/env python3
# Hypothesis: 4h Williams %R (14) extreme + 12h EMA50 trend filter + volume spike confirmation.
# Williams %R below -80 = oversold (long setup), above -20 = overbought (short setup).
# Enter long when %R crosses above -80 from below AND price > 12h EMA50 AND volume > 2.0x 20-bar average.
# Enter short when %R crosses below -20 from above AND price < 12h EMA50 AND volume > 2.0x 20-bar average.
# Exit when %R crosses above -20 (long) or below -80 (short) OR trend reversal.
# Uses 4h primary for optimal trade frequency (target: 75-200/4 years), 12h HTF for trend alignment.
# Williams %R captures momentum extremes; volume spike confirms institutional interest; 12h EMA50 filters counter-trend trades.
# Designed to work in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend).

name = "4h_WilliamsR_12hEMA_Volume_v1"
timeframe = "4h"
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
    
    # Get 4h data for Williams %R calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Williams %R(14) on 4h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current 4h volume > 2.0x 20-period average
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter_4h = volume_4h > (2.0 * vol_ma_4h)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (from below) AND price > 12h EMA50 AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema50_12h_aligned[i] and volume_filter_4h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (from above) AND price < 12h EMA50 AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema50_12h_aligned[i] and volume_filter_4h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) OR trend reversal (price < 12h EMA50)
            if williams_r[i] > -20 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) OR trend reversal (price > 12h EMA50)
            if williams_r[i] < -80 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals