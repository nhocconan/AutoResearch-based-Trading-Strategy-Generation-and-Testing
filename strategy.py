#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Williams %R extreme reversal with 6h EMA34 trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 6h EMA34 AND volume > 1.8 * avg_volume(20) on 6h
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 6h EMA34 AND volume > 1.8 * avg_volume(20) on 6h
# Exit when Williams %R crosses back through -50 (mean reversion midpoint) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R provides timely reversal signals in ranging markets
# 6h EMA34 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms reversal strength and reduces false signals
# Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)

name = "6h_WilliamsR_EXT_6hEMA34_VolumeSpike"
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
    if len(df_12h) < 14:  # Need at least one completed 12h bar for Williams %R
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R(14) on 12h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)) * -100
    
    # Align Williams %R to 6h timeframe (wait for completed 12h bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Get 6h data ONCE before loop for EMA34 trend filter
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_6h = df_6h['close'].values
    
    # Calculate 6h EMA34
    close_6h_series = pd.Series(close_6h)
    ema34_6h = close_6h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h_aligned = align_htf_to_ltf(prices, df_6h, ema34_6h)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_6h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold), above 6h EMA34, volume confirmation, in session
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                close[i] > ema34_6h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought), below 6h EMA34, volume confirmation, in session
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  close[i] < ema34_6h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion) OR volume drops below average
            if williams_r_aligned[i] > -50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion) OR volume drops below average
            if williams_r_aligned[i] < -50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals