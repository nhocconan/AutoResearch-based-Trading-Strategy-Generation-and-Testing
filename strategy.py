#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d volume spike and 1w EMA34 trend filter
# Long when: Williams %R(14) crosses above -80 (oversold), volume > 2.0x 48-period average, and close > 1w EMA34
# Short when: Williams %R(14) crosses below -20 (overbought), volume > 2.0x 48-period average, and close < 1w EMA34
# Exit when Williams %R returns to -50 (mean reversion) or opposite extreme
# Uses Williams %R for timely reversals in ranging markets, volume spike for conviction, 1w EMA for major trend filter
# Timeframe: 4h, HTF: 1d/1w. Target: 50-150 total trades over 4 years (12-38/year) to avoid fee drag.

name = "4h_WilliamsR_Extreme_1wEMA34_1dVolumeSpike"
timeframe = "4h"
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
    
    # Calculate volume confirmation on 4h using 48-period MA (equivalent to 1d lookback)
    if len(volume) >= 48:
        vol_ma_48 = pd.Series(volume).rolling(window=48, min_periods=48).mean().values
        volume_filter = volume > (2.0 * vol_ma_48)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                         ((highest_high - close_1d) / (highest_high - lowest_low)) * -100, 
                         -50)  # default to neutral when range is zero
    
    # Align Williams %R and 1w EMA to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)  # already computed above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (oversold), volume filter, and above 1w EMA34
            if (williams_r_aligned[i] > -80 and 
                williams_r_aligned[i-1] <= -80 and  # crossover above -80
                volume_filter[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (overbought), volume filter, and below 1w EMA34
            elif (williams_r_aligned[i] < -20 and 
                  williams_r_aligned[i-1] >= -20 and  # crossover below -20
                  volume_filter[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (mean reversion) or crosses below -80 (stop)
            if williams_r_aligned[i] > -50 or williams_r_aligned[i] < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 (mean reversion) or crosses above -20 (stop)
            if williams_r_aligned[i] < -50 or williams_r_aligned[i] > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals