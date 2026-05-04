#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R breakout with 12h EMA34 trend filter and volume confirmation (>2.0x 20 EMA volume)
# Williams %R identifies overbought/oversold conditions - short when > -20 (overbought), long when < -80 (oversold)
# 12h EMA34 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters false signals (>2.0x average volume) - tighter to reduce trades to 12-37/year target
# Discrete sizing 0.25 minimizes fee churn while maintaining profitability
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in bull markets (bounce from oversold) and bear markets (reject at overbought)
# Focus on BTC/ETH by requiring 12h trend alignment (avoids SOL-only bias)

name = "6h_WilliamsR_12hEMA34_VolumeConfirm_v1"
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
    
    # Get 12h data for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough data for EMA34 calculation
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(34) trend filter from prior completed 12h bar
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_shifted = np.roll(ema_34_12h, 1)
    ema_34_12h_shifted[0] = np.nan
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_shifted)
    
    # Volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Williams %R(14) from prior completed 6h bar
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Shift by 1 to use only completed bar
    highest_high_14_shifted = np.roll(highest_high_14, 1)
    lowest_low_14_shifted = np.roll(lowest_low_14, 1)
    highest_high_14_shifted[0] = np.nan
    lowest_low_14_shifted[0] = np.nan
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = ((highest_high_14_shifted - close) / (highest_high_14_shifted - lowest_low_14_shifted)) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold) AND price > 12h EMA34 AND volume spike
            if williams_r[i] < -80.0 and close[i] > ema_34_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought) AND price < 12h EMA34 AND volume spike
            elif williams_r[i] > -20.0 and close[i] < ema_34_12h_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R > -20 (overbought) OR price crosses below 12h EMA34
            if williams_r[i] > -20.0 or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R < -80 (oversold) OR price crosses above 12h EMA34
            if williams_r[i] < -80.0 or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals