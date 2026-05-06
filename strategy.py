#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extremes with 1d EMA34 trend filter and volume confirmation
# Long when 12h Williams %R < -80 (oversold) AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 1.5 * avg_volume(20) on 6h
# Short when 12h Williams %R > -20 (overbought) AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 1.5 * avg_volume(20) on 6h
# Exit when 12h Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 1d EMA34 ensures we trade with the dominant daily trend filter
# Volume confirmation validates reversal strength while limiting false signals

name = "6h_WilliamsR_Extreme_1dEMA34_Trend_Volume"
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
    
    # Calculate 12h Williams %R (14-period)
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align 12h Williams %R to 6h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h Williams %R < -80 (oversold), 1d EMA34 > EMA34 previous (uptrend), volume spike, in session
            if (williams_r_aligned[i] < -80 and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h Williams %R > -20 (overbought), 1d EMA34 < EMA34 previous (downtrend), volume spike, in session
            elif (williams_r_aligned[i] > -20 and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 12h Williams %R crosses back above -50 (recovering from oversold)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 12h Williams %R crosses back below -50 (recovering from overbought)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals