#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian Channel breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian upper band AND ATR(14) > 1.5x 50-period average ATR AND volume > 1.5x 20-period average volume
# Short when price breaks below 20-period Donchian lower band AND ATR(14) > 1.5x 50-period average ATR AND volume > 1.5x 20-period average volume
# Exit when price crosses back inside the Donchian Channel (opposite band)
# Uses Donchian channels to capture breakouts, ATR filter to ensure volatility regime, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian Channel on 12h (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_avg = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (50 for ATR average + buffer)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(atr_14[i]) or np.isnan(atr_50_avg[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_current = atr_14[i]
        atr_threshold = atr_50_avg[i] * 1.5
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Get ATR values aligned to 12h timeframe
        atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
        atr_50_avg_aligned = align_htf_to_ltf(prices, df_1d, atr_50_avg)
        
        atr_current_aligned = atr_14_aligned[i]
        atr_threshold_aligned = atr_50_avg_aligned[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + ATR filter + volume confirmation
            if (price > upper_donchian[i] and atr_current_aligned > atr_threshold_aligned and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below lower Donchian + ATR filter + volume confirmation
            elif (price < lower_donchian[i] and atr_current_aligned > atr_threshold_aligned and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside Donchian Channel (below lower band)
            if price < lower_donchian[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside Donchian Channel (above upper band)
            if price > upper_donchian[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_1dATR_Volume"
timeframe = "12h"
leverage = 1.0