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
    
    # Calculate 6h ATR(14) for volatility normalization
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.concatenate([[close[0]], close[:-1]])))
    tr3 = pd.Series(np.abs(low - np.concatenate([[close[0]], close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d ATR(14) for HTF volatility filter
    tr1_d = pd.Series(daily_high - daily_low)
    tr2_d = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3_d = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr_d = pd.concat([tr1_d, tr2_d, tr3_d], axis=1).max(axis=1)
    atr_14_d = tr_d.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d EMA(200) for long-term trend filter
    ema_200_d = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_d)
    ema_200_6h = align_htf_to_ltf(prices, df_1d, ema_200_d)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14_d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(ema_200_6h[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 6h ATR is elevated (> 1.5x daily ATR)
        vol_regime = atr_14[i] > (1.5 * atr_14_6h[i])
        
        # Long conditions: 
        # 1. Price above both EMAs (uptrend alignment)
        # 2. Donchian breakout with volume confirmation
        # 3. Elevated volatility regime (avoid chop)
        if (close[i] > ema_50_6h[i] and 
            close[i] > ema_200_6h[i] and 
            close[i] > highest_20[i] and 
            volume_ratio[i] > 1.8 and 
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below both EMAs (downtrend alignment)
        # 2. Donchian breakdown with volume confirmation
        # 3. Elevated volatility regime
        elif (close[i] < ema_50_6h[i] and 
              close[i] < ema_200_6h[i] and 
              close[i] < lowest_20[i] and 
              volume_ratio[i] > 1.8 and 
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_DualEMA_Donchian_Volume_Regime"
timeframe = "6h"
leverage = 1.0