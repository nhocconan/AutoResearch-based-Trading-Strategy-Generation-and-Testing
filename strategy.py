#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian breakout with volume confirmation and 1d EMA trend filter
# - Uses 12h Donchian channels (20-period) for medium-term structure
# - Uses 4h volume spike for entry confirmation
# - Uses 1d EMA(34) for long-term trend direction
# - Enters long when price breaks above 12h Donchian upper band with volume and 1d EMA up
# - Enters short when price breaks below 12h Donchian lower band with volume and 1d EMA down
# - Exits when price returns to 12h Donchian middle (median) or opposite band
# - Designed to capture medium-term trend moves with institutional level respect
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_12hDonchian_20_Volume_EMA_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian upper and lower bands
    upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2  # Median line for exit
    
    # Align 12h Donchian channels to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_12h, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_12h, lower_20)
    middle_20_4h = align_htf_to_ltf(prices, df_12h, middle_20)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter (4h timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (1.8 * vol_ma_10)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(middle_20_4h[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_34_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 12h Donchian upper with volume and 1d EMA up
            if close[i] > upper_20_4h[i] and volume_spike[i] and close[i] > ema_34_1d_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 12h Donchian lower with volume and 1d EMA down
            elif close[i] < lower_20_4h[i] and volume_spike[i] and close[i] < ema_34_1d_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle OR breaks below lower band
            if close[i] < middle_20_4h[i] or close[i] < lower_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle OR breaks above upper band
            if close[i] > middle_20_4h[i] or close[i] > upper_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals