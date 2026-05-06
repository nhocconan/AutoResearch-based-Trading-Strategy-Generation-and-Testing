#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Williams %R for mean reversion with volume confirmation and weekly trend filter
# - Uses 1d Williams %R (14-period) for oversold/overbought conditions
# - Uses 1w EMA50 for long-term trend direction filter
# - Uses 12h volume spike for entry confirmation
# - Enters long when Williams %R < -80 (oversold) with volume and weekly uptrend
# - Enters short when Williams %R > -20 (overbought) with volume and weekly downtrend
# - Exits when Williams %R returns to -50 (neutral) or opposite extreme
# - Designed to capture mean reversion swings within the weekly trend
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_1dWilliamsR_1wEMA50_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close_1d) / (highest_high - lowest_low)) * -100,
        -50  # neutral when range is zero
    )
    
    # Align 1d Williams %R to 12h timeframe
    williams_r_12h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_1w_12h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter (12h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.5 * vol_ma_10)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_12h[i]) or np.isnan(ema_50_1w_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) with volume and weekly uptrend
            if williams_r_12h[i] < -80 and volume_spike[i] and close[i] > ema_50_1w_12h[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) with volume and weekly downtrend
            elif williams_r_12h[i] > -20 and volume_spike[i] and close[i] < ema_50_1w_12h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral (-50) or becomes overbought
            if williams_r_12h[i] >= -50 or williams_r_12h[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral (-50) or becomes oversold
            if williams_r_12h[i] <= -50 or williams_r_12h[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals