#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above 20-period Donchian upper channel AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period average
# Short when price breaks below 20-period Donchian lower channel AND ATR(14) > 1.5x ATR(50) AND volume > 1.5x 20-period average
# Exit when price crosses back to the middle of the Donchian channel
# Uses Donchian channels to capture breakouts, ATR ratio to filter for volatile breakouts, volume for confirmation
# Designed for 4h timeframe with ~20-50 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian Channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    middle_channel = (upper_channel + lower_channel) / 2
    
    # Calculate ATR(14) and ATR(50) on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Calculate ATR ratio (ATR14/ATR50) for volatility expansion filter
    atr_ratio = np.divide(atr14, atr50, out=np.zeros_like(atr14), where=atr50!=0)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (50 for ATR50 + buffer)
    start = 60
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: break above upper Donchian + volatility expansion + volume confirmation
            if (price > upper_channel[i] and atr_ratio_aligned[i] > 1.5 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: break below lower Donchian + volatility expansion + volume confirmation
            elif (price < lower_channel[i] and atr_ratio_aligned[i] > 1.5 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle of Donchian channel
            if price < middle_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle of Donchian channel
            if price > middle_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_1dATR_Volume"
timeframe = "4h"
leverage = 1.0