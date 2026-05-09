#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
# Uses Williams %R (14) for mean reversion signals, daily EMA34 for trend direction,
# and volume surge for confirmation. Designed to capture reversals in both bull and bear markets.
# Target: 15-35 trades/year to avoid fee drag while maintaining edge.
name = "12h_WilliamsR_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA for daily timeframe
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Williams %R (14-period) for 12h timeframe
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    williams_r = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i < 13:  # Need 14 periods (0-13 inclusive)
            williams_r[i] = np.nan
        else:
            highest_high_14 = np.max(high[i-13:i+1])
            lowest_low_14 = np.min(low[i-13:i+1])
            if highest_high_14 != lowest_low_14:
                williams_r[i] = -100 * (highest_high_14 - close[i]) / (highest_high_14 - lowest_low_14)
            else:
                williams_r[i] = -50  # Avoid division by zero
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period EMA (moderate threshold)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.8 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Need 13 periods for Williams %R
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + price above EMA34 + volume spike
            if (williams_r[i] < -80 and price > ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + price below EMA34 + volume spike
            elif (williams_r[i] > -20 and price < ema_34_1d_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns above -50 or price crosses below EMA34
            if williams_r[i] > -50 or price < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns below -50 or price crosses above EMA34
            if williams_r[i] < -50 or price > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals