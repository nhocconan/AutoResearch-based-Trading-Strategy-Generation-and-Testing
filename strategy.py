#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness index regime filter + Donchian(20) breakout with volume confirmation
# Long when price breaks above Donchian(20) high AND Choppiness(14) < 38.2 (trending regime) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND Choppiness(14) < 38.2 AND volume > 1.5x 20-period average
# Exit when price crosses back inside the Donchian channel (opposite band) OR Choppiness > 61.8 (range regime)
# Uses regime filter to avoid whipsaws in sideways markets, focusing on trending moves
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels on 4h (20-period high/low)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate True Range for Choppiness index
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(14) for Choppiness
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate max(high) and min(low) over 14 periods for Choppiness
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness index: 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    range14 = max_high14 - min_low14
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    chop = 100 * np.log10(sum_atr14 / range14) / np.log10(14)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (max of 20 for Donchian, 14*2 for Chop)
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(chop[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: breakout above Donchian high + trending regime (CHOP < 38.2) + volume confirmation
            if (price > high_20[i] and chop[i] < 38.2 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Donchian low + trending regime + volume confirmation
            elif (price < low_20[i] and chop[i] < 38.2 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Donchian low OR choppy regime (CHOP > 61.8)
            if price < low_20[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Donchian high OR choppy regime
            if price > high_20[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_Volume_Regime"
timeframe = "4h"
leverage = 1.0