#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d EMA34 trend filter and volume spike
# Long when: Williams %R(14) < -80 (oversold), volume > 2x 20-period average, and close > 1d EMA34
# Short when: Williams %R(14) > -20 (overbought), volume > 2x 20-period average, and close < 1d EMA34
# Exit when: Williams %R returns to -50 level (mean reversion) or opposite extreme
# Williams %R identifies exhaustion points; EMA34 filters trend direction; volume confirms conviction
# Effective in ranging markets (mean reversion from extremes) and can catch reversals in trends
# Timeframe: 6h, HTF: 1d. Target: 75-150 total trades over 4 years (19-38/year).

name = "6h_WilliamsRExtreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 6h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for Williams %R and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R on 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    if len(high_1d) >= 14:
        highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(len(close_1d), np.nan)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R < -80 (oversold), volume filter, and above 1d EMA34
            if (williams_r_aligned[i] < -80 and 
                volume_filter[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R > -20 (overbought), volume filter, and below 1d EMA34
            elif (williams_r_aligned[i] > -20 and 
                  volume_filter[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to -50 (mean reversion from oversold)
            if williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to -50 (mean reversion from overbought)
            if williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals