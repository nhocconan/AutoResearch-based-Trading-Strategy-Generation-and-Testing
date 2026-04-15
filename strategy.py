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
    
    # Calculate weekly EMA(34) for trend filter
    ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly Donchian channels (20-period) for breakout levels
    weekly_highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_34_6h = align_htf_to_ltf(prices, df_1w, ema_34)
    atr_14_6h = align_htf_to_ltf(prices, df_1w, atr_14)
    weekly_highest_20_6h = align_htf_to_ltf(prices, df_1w, weekly_highest_20)
    weekly_lowest_20_6h = align_htf_to_ltf(prices, df_1w, weekly_lowest_20)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(weekly_highest_20_6h[i]) or np.isnan(weekly_lowest_20_6h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA34
        # 2. 6h Donchian breakout/breakdown in direction of weekly trend
        # 3. Weekly volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. 6h volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: Weekly uptrend + 6h Donchian breakout above weekly high
        if (close[i] > ema_34_6h[i] and              # Weekly uptrend filter
            close[i] > highest_20[i] and             # 6h Donchian breakout
            close[i] > weekly_highest_20_6h[i] and   # Break above weekly Donchian high
            volume_ratio[i] > 1.3 and                # Volume confirmation
            atr_14_6h[i] > 0.005 * close[i]):        # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Weekly downtrend + 6h Donchian breakdown below weekly low
        elif (close[i] < ema_34_6h[i] and            # Weekly downtrend filter
              close[i] < lowest_20[i] and            # 6h Donchian breakdown
              close[i] < weekly_lowest_20_6h[i] and  # Break below weekly Donchian low
              volume_ratio[i] > 1.3 and              # Volume confirmation
              atr_14_6h[i] > 0.005 * close[i]):      # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA34_Donchian20_Breakout_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0