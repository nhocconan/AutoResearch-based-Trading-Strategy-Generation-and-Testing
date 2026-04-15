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
    daily_volume = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter (using true range)
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr_daily = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_daily = pd.Series(tr_daily).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20_daily = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_daily = daily_volume / (vol_ma_20_daily + 1e-10)
    
    # Align HTF ATR and volume ratio to 4h timeframe
    atr_14_4h = align_htf_to_ltf(prices, df_1d, atr_14_daily)
    volume_ratio_4h = align_htf_to_ltf(prices, df_1d, volume_ratio_daily)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_4h[i]) or np.isnan(volume_ratio_4h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 4h price breaks above 20-period Donchian high with volume confirmation → long
        # 2. 4h price breaks below 20-period Donchian low with volume confirmation → short
        # 3. Volatility filter: ATR > 0.8% of price (ensure sufficient volatility)
        # 4. Volume confirmation: volume > 1.4x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 4h breakout above Donchian high
        if (close[i] > highest_20[i] and            # 4h price above Donchian high
            volume_ratio_4h[i] > 1.4 and           # Volume confirmation
            atr_14_4h[i] > 0.008 * close[i]):      # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 4h breakdown below Donchian low
        elif (close[i] < lowest_20[i] and          # 4h price below Donchian low
              volume_ratio_4h[i] > 1.4 and         # Volume confirmation
              atr_14_4h[i] > 0.008 * close[i]):    # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0