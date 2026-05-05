#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d volume spike and 1d EMA34 trend filter
# Long when: Williams %R < -80 (oversold), volume > 1.5x 20-period average, and close > 1d EMA34
# Short when: Williams %R > -20 (overbought), volume > 1.5x 20-period average, and close < 1d EMA34
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme
# Williams %R identifies exhaustion points; volume spike confirms conviction; 1d EMA34 filters counter-trend
# Effective in ranging markets (mean reversion from extremes) and can catch reversals in trends
# Timeframe: 6h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsRExtreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 6h (14-period)
    lookback = 14
    if len(high) >= lookback:
        highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
        lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
        # Avoid division by zero
        hl_range = highest_high - lowest_low
        hl_range = np.where(hl_range == 0, 1e-10, hl_range)
        willr = -100 * (highest_high - close) / hl_range
    else:
        willr = np.full(n, np.nan)
    
    # Volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(willr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R oversold (< -80), volume filter, and above 1d EMA34
            if (willr[i] < -80 and 
                volume_filter[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R overbought (> -20), volume filter, and below 1d EMA34
            elif (willr[i] > -20 and 
                  volume_filter[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion) or breaks above -20 (reversal)
            if willr[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion) or breaks below -80 (reversal)
            if willr[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals