#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter with 12-hour EMA trend and volume confirmation
# Long when: Choppiness Index > 61.8 (ranging market), price crosses above EMA20, volume > 1.5x average
# Short when: Choppiness Index > 61.8 (ranging market), price crosses below EMA20, volume > 1.5x average
# Exit when: Choppiness Index < 38.2 (trending market) OR price crosses EMA20 in opposite direction
# Uses Choppiness Index to identify ranging markets where mean reversion works, EMA for dynamic support/resistance
# Target: 80-150 total trades over 4 years (20-38/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA20 for dynamic support/resistance
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period)
    atr_list = []
    for i in range(n):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_list.append(tr)
    
    atr = pd.Series(atr_list).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_raw = 100 * np.log10((highest_high - lowest_low) / (np.sum(atr_list[:14]) if i < 14 else np.sum(atr_list[i-13:i+1])) / 14) / np.log10(14)
    # Simplified calculation using rolling sum
    atr_sum = pd.Series(atr_list).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10((highest_high - lowest_low) / atr_sum) / np.log10(14)
    # Handle cases where highest_high == lowest_low
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: ranging market (CHOP > 61.8), price crosses above EMA20, volume confirmation
            if (chop[i] > 61.8 and price > ema20[i] and 
                i > start and close[i-1] <= ema20[i-1] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: ranging market (CHOP > 61.8), price crosses below EMA20, volume confirmation
            elif (chop[i] > 61.8 and price < ema20[i] and 
                  i > start and close[i-1] >= ema20[i-1] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trending market (CHOP < 38.2) OR price crosses below EMA20
            if chop[i] < 38.2 or (price < ema20[i] and close[i-1] >= ema20[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trending market (CHOP < 38.2) OR price crosses above EMA20
            if chop[i] < 38.2 or (price > ema20[i] and close[i-1] <= ema20[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Choppiness_EMA20_Volume"
timeframe = "4h"
leverage = 1.0