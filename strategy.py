#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly Supertrend (ATR=10, mult=3)
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    hl2 = (weekly_high + weekly_low) / 2.0
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    supertrend = np.zeros_like(weekly_close)
    direction = np.ones_like(weekly_close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(weekly_close)):
        # Upper band logic
        if weekly_close[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        # Lower band logic
        if weekly_close[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
            
        # Supertrend logic
        if supertrend[i-1] == upper_band[i-1]:
            if weekly_close[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if weekly_close[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    # Calculate weekly Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 1d timeframe with proper delay
    supertrend_1d = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_1d = align_htf_to_ltf(prices, df_1w, direction)
    highest_20_1d = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_1d = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1_1d = pd.Series(high - low)
    tr2_1d = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3_1d = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    atr_14_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = volume / (vol_ma_20_1d + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_1d[i]) or np.isnan(direction_1d[i]) or 
            np.isnan(highest_20_1d[i]) or np.isnan(lowest_20_1d[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(volume_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: Weekly uptrend + price breaks above weekly Donchian high + volume confirmation
        # Short: Weekly downtrend + price breaks below weekly Donchian low + volume confirmation
        
        # Long conditions
        if (direction_1d[i] == 1 and              # Weekly uptrend
            close[i] > highest_20_1d[i] and       # Price breaks above weekly Donchian high
            volume_ratio_1d[i] > 1.5 and          # Volume confirmation
            atr_14_1d[i] > 0.01 * close[i]):      # Volatility filter (avoid low volatility)
            signals[i] = 0.25
            
        # Short conditions
        elif (direction_1d[i] == -1 and           # Weekly downtrend
              close[i] < lowest_20_1d[i] and      # Price breaks below weekly Donchian low
              volume_ratio_1d[i] > 1.5 and        # Volume confirmation
              atr_14_1d[i] > 0.01 * close[i]):    # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Supertrend_Weekly_Donchian20_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0