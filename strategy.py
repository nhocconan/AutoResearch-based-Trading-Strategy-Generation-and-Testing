#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using weekly Donchian breakout with volume confirmation and volatility filter
# - Uses 1w Donchian channels (20-period) for long-term structure and major trend identification
# - Uses 4h volume spike (1.5x 20-period MA) for entry confirmation to avoid false breakouts
# - Uses 4h ATR-based volatility filter (ATR < 50-day ATR median) to avoid choppy markets
# - Enters long when price breaks above 1w Donchian upper band with volume and low volatility
# - Enters short when price breaks below 1w Donchian lower band with volume and low volatility
# - Exits when price returns to 1w Donchian middle (average of upper/lower) or opposite band
# - Designed to capture major trend moves with reduced false signals in ranging markets
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "4h_1wDonchian_20_Volume_VolatilityFilter"
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
    
    # Get 1w data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian upper and lower bands
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2  # Middle line for exit
    
    # Align 1w Donchian channels to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1w, lower_20)
    middle_20_4h = align_htf_to_ltf(prices, df_1w, middle_20)
    
    # Volume filter (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    # Volatility filter (4h timeframe) - avoid choppy markets
    # Calculate ATR(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 50-period ATR median for volatility regime filter
    atr_median = pd.Series(atr).rolling(window=50, min_periods=50).median().values
    low_volatility = atr < atr_median  # Only trade when volatility is below median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(middle_20_4h[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(low_volatility[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 1w Donchian upper with volume and low volatility
            if close[i] > upper_20_4h[i] and volume_spike[i] and low_volatility[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below 1w Donchian lower with volume and low volatility
            elif close[i] < lower_20_4h[i] and volume_spike[i] and low_volatility[i]:
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