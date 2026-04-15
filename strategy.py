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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = pivot + 1.1 * (daily_high - daily_low) / 12.0
    s1 = pivot - 1.1 * (daily_high - daily_low) / 12.0
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe with proper delay
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 4h Donchian channels (20-period) for breakout confirmation
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 4h close above 1d R1 + volume confirmation + volatility filter + price above 20-period high
        # Short: 4h close below 1d S1 + volume confirmation + volatility filter + price below 20-period low
        # Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > r1_4h[i] and              # 4h close above 1d R1
            close[i] > highest_20[i] and         # Price above 20-period high (breakout confirmation)
            volume_ratio[i] > 1.5 and            # Strong volume confirmation
            atr_14_4h[i] > 0.003 * close[i]):    # Volatility filter (avoid low volatility chop)
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < s1_4h[i] and            # 4h close below 1d S1
              close[i] < lowest_20[i] and        # Price below 20-period low (breakdown confirmation)
              volume_ratio[i] > 1.5 and          # Strong volume confirmation
              atr_14_4h[i] > 0.003 * close[i]):  # Volatility filter (avoid low volatility chop)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Donchian_Volume_Filter"
timeframe = "4h"
leverage = 1.0