#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extremes with 1d EMA34 trend filter and volume confirmation
# Long when 12h Williams %R < -80 (oversold) AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 1.5 * avg_volume(20) on 6h
# Short when 12h Williams %R > -20 (overbought) AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when 12h Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe
# Williams %R on 12h provides reliable mean-reversion signals in ranging markets
# 1d EMA34 ensures we trade with the dominant higher timeframe trend
# Volume confirmation filters out low-momentum false signals
# Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets

name = "6h_12hWilliamsR_Extreme_1dEMA34_Trend_Volume"
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
    
    # Get 12h data ONCE before loop for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:  # Need at least 14 completed 12h bars for Williams %R
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)) * -100
    
    # Align 12h Williams %R to 6h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed 1d bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h Williams %R < -80 (oversold), 1d EMA34 > EMA34 previous (uptrend), volume spike
            if (williams_r_aligned[i] < -80 and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h Williams %R > -20 (overbought), 1d EMA34 < EMA34 previous (downtrend), volume spike
            elif (williams_r_aligned[i] > -20 and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 12h Williams %R crosses above -50 (momentum fading)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 12h Williams %R crosses below -50 (momentum fading)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals