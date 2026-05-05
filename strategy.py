#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA34 trend filter and Bollinger Bands squeeze confirmation
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# 1d EMA34 determines trend: only long when price > 1d EMA34, short when price < 1d EMA34
# Bollinger Bands squeeze (BBW < 0.05) on 6h filters for low volatility breakout setups
# Exit when Williams %R reverts to midpoint (-50) or opposite extreme
# Combines mean reversion (Williams %R) with trend filter (1d EMA) and volatility contraction expansion pattern
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_WilliamsR_1dEMA34_BBSqueeze"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need sufficient data for EMA34
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 6h
    if len(high) >= 14 and len(low) >= 14 and len(close) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, -50)
    
    # Bollinger Bands squeeze on 6h (20, 2)
    if len(close) >= 20:
        ma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
        std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
        upper_bb = ma_20 + 2 * std_20
        lower_bb = ma_20 - 2 * std_20
        bb_width = (upper_bb - lower_bb) / ma_20
        bb_squeeze = bb_width < 0.05  # Squeeze when BB width < 5%
    else:
        ma_20 = np.full(n, np.nan)
        bb_squeeze = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(ma_20[i]) if i < len(ma_20) else False):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80) AND above 1d EMA34 AND BB squeeze
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                bb_squeeze[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20) AND below 1d EMA34 AND BB squeeze
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  bb_squeeze[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reverts to midpoint (-50) or becomes overbought
            if williams_r[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reverts to midpoint (-50) or becomes oversold
            if williams_r[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals