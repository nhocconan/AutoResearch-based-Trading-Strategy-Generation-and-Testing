#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands with volume confirmation
# - Uses 1w Bollinger Bands (20, 2.0) for dynamic support/resistance
# - Uses 1d volume spike for entry confirmation
# - Enters long when price breaks above upper band with volume spike
# - Enters short when price breaks below lower band with volume spike
# - Exits when price returns to middle band
# - Designed to capture weekly trends with statistical significance
# - Target: 30-100 total trades over 4 years (7-25/year) with 0.25 position sizing

name = "1d_1wBBands_1dVolume_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Bollinger Bands (20, 2.0)
    close_1w = df_1w['close'].values
    sma_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    
    upper_band = sma_20 + (2.0 * std_20)
    middle_band = sma_20
    lower_band = sma_20 - (2.0 * std_20)
    
    # Align 1w Bollinger Bands to 1d timeframe
    upper_1d = align_htf_to_ltf(prices, df_1w, upper_band)
    middle_1d = align_htf_to_ltf(prices, df_1w, middle_band)
    lower_1d = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume filter (1d timeframe)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_spike = volume > (2.0 * vol_ma_10)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_1d[i]) or np.isnan(middle_1d[i]) or 
            np.isnan(lower_1d[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band with volume spike
            if close[i] > upper_1d[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with volume spike
            elif close[i] < lower_1d[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle band
            if close[i] < middle_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle band
            if close[i] > middle_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals