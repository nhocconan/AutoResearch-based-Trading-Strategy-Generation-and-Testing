#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d trend filter
# - Uses 4h Donchian channels (20-period) for medium-term structure
# - Uses 1h volume spike for entry confirmation
# - Uses 1d EMA50 for trend filter (only long when price > EMA50, short when price < EMA50)
# - Designed to capture trend moves with institutional level respect
# - Target: 60-150 total trades over 4 years (15-37/year) with 0.20 position sizing

name = "1h_4hDonchian_20_1dEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian channels to 1h timeframe
    upper_20_1h = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_1h = align_htf_to_ltf(prices, df_4h, lower_20)
    
    # Volume filter (1h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.8 * vol_ma_10)  # Strong volume confirmation
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_1h[i]) or np.isnan(lower_20_1h[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_50_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h Donchian upper with volume and uptrend (price > 1d EMA50)
            if close[i] > upper_20_1h[i] and volume_spike[i] and close[i] > ema_50_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h Donchian lower with volume and downtrend (price < 1d EMA50)
            elif close[i] < lower_20_1h[i] and volume_spike[i] and close[i] < ema_50_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns below 4h Donchian lower
            if close[i] < lower_20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns above 4h Donchian upper
            if close[i] > upper_20_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals