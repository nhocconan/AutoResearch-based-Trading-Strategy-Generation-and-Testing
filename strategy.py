#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Williams %R extreme levels with volume confirmation
# Long when 12h Williams %R < -80 (oversold) AND price > 12h EMA34 AND volume > 1.3 * avg_volume(20)
# Short when 12h Williams %R > -20 (overbought) AND price < 12h EMA34 AND volume > 1.3 * avg_volume(20)
# Exit when price crosses 12h EMA34 in opposite direction or Williams %R returns to -50
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Williams %R identifies overextended moves; EMA34 filters for trend alignment
# Volume confirmation ensures breakout validity
# Works in bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "4h_12hWilliamsR_Extreme_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Williams %R and EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 35:  # Need enough data for EMA34 and Williams %R
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14) * -100
    williams_r_12h = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_12h)
    
    # Calculate 12h EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h indicators to 4h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price > EMA34 AND volume confirmation
            if (williams_r_aligned[i] < -80 and close[i] > ema_34_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price < EMA34 AND volume confirmation
            elif (williams_r_aligned[i] > -20 and close[i] < ema_34_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below EMA34 OR Williams %R returns to -50 (mean reversion)
            if close[i] < ema_34_aligned[i] or williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above EMA34 OR Williams %R returns to -50 (mean reversion)
            if close[i] > ema_34_aligned[i] or williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals