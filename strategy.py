#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with daily ATR breakout and volume confirmation
# Long when price breaks above daily ATR-based upper band AND Choppiness Index < 38.2 (trending) AND volume > 1.5x 20-period average
# Short when price breaks below daily ATR-based lower band AND Choppiness Index < 38.2 (trending) AND volume > 1.5x 20-period average
# Exit when price crosses back inside the daily ATR-based channel
# Uses daily ATR to adapt volatility regime, Choppiness Index to filter choppy markets, volume for confirmation
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR on 1d (14-period)
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
    
    # Calculate Choppiness Index on 1d (14-period)
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop = np.full_like(atr_14, 50.0)  # Default to neutral
    mask = (highest_high - lowest_low) != 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / (highest_high[mask] - lowest_low[mask])) / np.log10(14)
    
    # Calculate ATR-based channel on 1d (using 1x ATR for breakout sensitivity)
    upper_channel = pd.Series(close_1d).rolling(window=14, min_periods=14).mean().values + atr_14
    lower_channel = pd.Series(close_1d).rolling(window=14, min_periods=14).mean().values - atr_14
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_value = chop_aligned[i]
        atr_value = atr_14_aligned[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        # Only trade in trending markets (Choppiness Index < 38.2)
        if chop_value >= 38.2:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long setup: price breaks above upper channel + volume confirmation
            if price > upper_channel_aligned[i] and vol > vol_threshold:
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below lower channel + volume confirmation
            elif price < lower_channel_aligned[i] and vol > vol_threshold:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back inside channel (below lower band)
            if price < lower_channel_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back inside channel (above upper band)
            if price > upper_channel_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Choppiness_ATR_Channel_Volume"
timeframe = "4h"
leverage = 1.0