#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme levels with 1d trend filter and volume confirmation.
Long when Williams %R crosses above -20 from below (oversold reversal) with bullish 1d trend and volume spike.
Short when Williams %R crosses below -80 from above (overbought reversal) with bearish 1d trend and volume spike.
Exit when Williams %R returns to -50 (mean reversion).
Uses 1d EMA34 for trend filter to capture medium-term trend and avoid whipsaws.
Designed for low trade frequency (20-40/year) to minimize fee drift.
Williams %R is effective in both bull and bear markets as it identifies reversal points at extremes.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 14-period Williams %R
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    # Williams %R values for previous bar (to detect crosses)
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = -50  # neutral start
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after Williams %R lookback
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -20 from below (oversold reversal)
            # with bullish 1d trend and volume spike
            if (williams_r[i] > -20 and williams_r_prev[i] <= -20 and
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -80 from above (overbought reversal)
            # with bearish 1d trend and volume spike
            elif (williams_r[i] < -80 and williams_r_prev[i] >= -80 and
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to -50 (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses below -50 from above
                if williams_r[i] < -50 and williams_r_prev[i] >= -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses above -50 from below
                if williams_r[i] > -50 and williams_r_prev[i] <= -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_Extreme_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0
#%%