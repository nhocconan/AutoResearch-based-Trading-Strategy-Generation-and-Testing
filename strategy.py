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
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]]))
    tr3 = np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(weekly_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = weekly_volume / (vol_ma_20 + 1e-10)
    
    # Calculate weekly Donchian(20) channels
    highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian channels and volatility/volume filters to 4h timeframe
    highest_20_4h = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_4h = align_htf_to_ltf(prices, df_1w, lowest_20)
    atr_14_4h = align_htf_to_ltf(prices, df_1w, atr_14)
    volume_ratio_4h = align_htf_to_ltf(prices, df_1w, volume_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_4h[i]) or np.isnan(lowest_20_4h[i]) or 
            np.isnan(atr_14_4h[i]) or np.isnan(volume_ratio_4h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above weekly Donchian high with volume confirmation → long
        # 2. 4h price breaks below weekly Donchian low with volume confirmation → short
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above weekly Donchian high
        if (close[i] > highest_20_4h[i] and            # 4h price above weekly Donchian high
            volume_ratio_4h[i] > 1.5 and               # Volume confirmation
            atr_14_4h[i] > 0.003 * close[i]):          # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below weekly Donchian low
        elif (close[i] < lowest_20_4h[i] and           # 4h price below weekly Donchian low
              volume_ratio_4h[i] > 1.5 and             # Volume confirmation
              atr_14_4h[i] > 0.003 * close[i]):        # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_WeeklyDonchian20_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0