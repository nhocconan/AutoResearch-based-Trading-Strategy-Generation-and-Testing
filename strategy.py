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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily Donchian channels (20-period)
    upper_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 4h timeframe
    upper_20_4h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_4h = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Calculate 4h ATR(14) for stoploss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_4h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_4h[i]) or np.isnan(lower_20_4h[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above 1d Donchian upper with volume confirmation → long
        # 2. 4h price breaks below 1d Donchian lower with volume confirmation → short
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above 1d Donchian upper
        if (close[i] > upper_20_4h[i] and            # 4h price above 1d Donchian upper
            volume_ratio[i] > 1.5):                  # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below 1d Donchian lower
        elif (close[i] < lower_20_4h[i] and          # 4h price below 1d Donchian lower
              volume_ratio[i] > 1.5):                # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0