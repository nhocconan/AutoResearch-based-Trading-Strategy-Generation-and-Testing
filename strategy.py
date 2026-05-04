#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) mean reversion with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; mean reversion works in ranging markets
# 1d EMA34 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (>1.3x 20 EMA) ensures reversal has participation
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h.
# Williams %R is effective in both bull and bear markets when combined with trend and volume filters.

name = "6h_WilliamsR14_1dEMA34_VolumeConfirm"
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Calculate Williams %R on 6h timeframe: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Rolling highest high and lowest low for Williams %R
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R calculation
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_6h) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align Williams %R to 6h timeframe (no additional delay needed as it's based on completed 6h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
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
            # Long conditions: Williams %R oversold (< -80) + price above 1d EMA34 + volume confirmation
            if williams_r_aligned[i] < -80 and close[i] > ema_34_1d_aligned[i] and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) + price below 1d EMA34 + volume confirmation
            elif williams_r_aligned[i] > -20 and close[i] < ema_34_1d_aligned[i] and volume[i] > (1.3 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) OR price crosses below 1d EMA34
            if williams_r_aligned[i] > -50 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) OR price crosses above 1d EMA34
            if williams_r_aligned[i] < -50 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals