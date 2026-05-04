#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# Uses Donchian channel from prior completed 1w bar for structure (strong trend following)
# 1w EMA(50) filter ensures we only trade in the direction of the weekly trend
# Volume confirmation (>1.5x 20 EMA volume) ensures breakout has participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 30-100 total trades over 4 years = 7-25/year for 1d timeframe
# Works in bull markets via breakouts and in bear markets via shorting breakdowns
# Weekly trend filter prevents counter-trend whipsaws

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter and Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1w EMA(50) trend filter from prior completed 1w bar
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Shift EMA by 1 to use only prior completed 1w bar (no look-ahead)
    ema_50_1w_shifted = np.roll(ema_50_1w, 1)
    ema_50_1w_shifted[0] = np.nan
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_shifted)
    
    # Calculate Donchian channels (20-period) from prior completed 1w bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high over past 20 weekly bars
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 weekly bars
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use only prior completed 1w bar
    upper_20_shifted = np.roll(upper_20, 1)
    lower_20_shifted = np.roll(lower_20, 1)
    upper_20_shifted[0] = np.nan
    lower_20_shifted[0] = np.nan
    
    # Align Donchian levels to 1d timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20_shifted)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + price > weekly EMA50 (uptrend) + volume spike
            if close[i] > upper_20_aligned[i] and close[i] > ema_50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + price < weekly EMA50 (downtrend) + volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < ema_50_1w_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly EMA50 OR Donchian middle band
            middle_band = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if not np.isnan(middle_band) and (close[i] < ema_50_1w_aligned[i] or close[i] < middle_band):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly EMA50 OR Donchian middle band
            middle_band = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if not np.isnan(middle_band) and (close[i] > ema_50_1w_aligned[i] or close[i] > middle_band):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals