#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) with 1d EMA34 trend filter and volume confirmation (>1.3x 20 EMA volume)
# Williams %R identifies overbought/oversold conditions: long when %R crosses above -80 from below,
# short when %R crosses below -20 from above. 1d EMA34 ensures alignment with higher timeframe trend
# to avoid counter-trend whipsaws. Volume confirmation filters false signals (>1.3x average volume).
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability.
# Works in bull markets (continuation from oversold) and bear markets (continuation from overbought).
# Focus on BTC/ETH by requiring 1d trend alignment (avoids SOL-only bias).

name = "6h_WilliamsR_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14) from prior completed 1d bar
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low + 1e-10) * -100
    williams_r_shifted = np.roll(williams_r, 1)
    williams_r_shifted[0] = np.nan
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_shifted)
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 AND price > 1d EMA34 AND volume spike
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                close[i] > ema_34_1d_aligned[i] and volume[i] > (1.3 * vol_ema_20[i])):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 AND price < 1d EMA34 AND volume spike
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  close[i] < ema_34_1d_aligned[i] and volume[i] > (1.3 * vol_ema_20[i])):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 OR price crosses below 1d EMA34
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 OR price crosses above 1d EMA34
            if (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals