#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA34 trend filter and volume spike confirmation
# Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Exit when Williams %R crosses back through -50 (mean reversion) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year)
# Williams %R identifies exhaustion points; 1d EMA34 filters primary trend; volume spike confirms reversal strength
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)

name = "6h_WilliamsR_Extreme_Reversal_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold reversal), above 1d EMA34, volume confirmation
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought reversal), below 1d EMA34, volume confirmation
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 OR volume drops below average
            if williams_r[i] > -50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 OR volume drops below average
            if williams_r[i] < -50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals