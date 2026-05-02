#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R Breakout with 4h EMA34 Trend Filter and Volume Confirmation
# Uses 4h Williams %R to identify overbought/oversold conditions in the context of 4h trend
# Entry logic: Long when Williams %R crosses above -80 from below (oversold bounce) in uptrend (price > 4h EMA34) with volume spike
#              Short when Williams %R crosses below -20 from above (overbought rejection) in downtrend (price < 4h EMA34) with volume spike
# Exit logic: Reverse position when opposite Williams %R extreme is reached or trend changes
# Works in both bull and bear markets by trading mean reversion within the 4h trend
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# Discrete sizing 0.20 balances profit potential and fee drag

name = "1h_WilliamsR_Breakout_4hEMA34_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_4h = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r_4h = (highest_high_4h - close_4h) / (highest_high_4h - lowest_low_4h) * -100
    # Handle division by zero (when high == low)
    williams_r_4h = np.where((highest_high_4h - lowest_low_4h) == 0, -50, williams_r_4h)
    
    # Align Williams %R to 1h timeframe (use previous 4h bar's values)
    williams_r_4h_aligned = align_htf_to_ltf(prices, df_4h, williams_r_4h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(williams_r_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (from below) AND price > 4h EMA34 (uptrend) AND volume spike
            if (williams_r_4h_aligned[i] > -80 and 
                williams_r_4h_aligned[i-1] <= -80 and  # crossed above -80
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Williams %R crosses below -20 (from above) AND price < 4h EMA34 (downtrend) AND volume spike
            elif (williams_r_4h_aligned[i] < -20 and 
                  williams_r_4h_aligned[i-1] >= -20 and  # crossed below -20
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R reaches overbought (-20) OR trend changes (price < 4h EMA34)
            if (williams_r_4h_aligned[i] >= -20 or 
                close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Williams %R reaches oversold (-80) OR trend changes (price > 4h EMA34)
            if (williams_r_4h_aligned[i] <= -80 or 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals