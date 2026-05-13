#!/usr/bin/env python3
# Hypothesis: 4h Williams %R mean reversion with 12h EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold conditions. Long when %R < -80 and rising, short when %R > -20 and falling.
# Uses 12h EMA50 to filter trend direction (avoid counter-trend trades). Volume > 1.5x average confirms institutional participation.
# Designed for low trade frequency (<50/year) to minimize fee drag. Discrete sizing 0.25.
# Works in bull markets via oversold bounces in uptrend, and in bear markets via overbought reversals in downtrend.

name = "4h_WilliamsR_MeanReversion_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (wait for 12h bar to close)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) and rising, price > 12h EMA50 (uptrend), volume > 1.5x average
            if (williams_r[i] < -80 and 
                i > 20 and williams_r[i] > williams_r[i-1] and  # Rising from oversold
                close[i] > ema50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) and falling, price < 12h EMA50 (downtrend), volume > 1.5x average
            elif (williams_r[i] > -20 and 
                  i > 20 and williams_r[i] < williams_r[i-1] and  # Falling from overbought
                  close[i] < ema50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (momentum weakening) OR price < 12h EMA50
            if williams_r[i] > -50 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (momentum weakening) OR price > 12h EMA50
            if williams_r[i] < -50 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals