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
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily volume MA(20) for volume filter
    vol_ma_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above 4h Donchian(20) high
        # 2. Volume confirmation: current volume > 1.5x daily average volume
        # 3. Volatility filter: ATR > 0.5% of price (avoid extremely low volatility)
        if (close[i] > donchian_high[i] and
            volume[i] > 1.5 * vol_ma_20_1d_aligned[i] and
            atr_14_1d_aligned[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 4h Donchian(20) low
        # 2. Volume confirmation: current volume > 1.5x daily average volume
        # 3. Volatility filter: ATR > 0.5% of price
        elif (close[i] < donchian_low[i] and
              volume[i] > 1.5 * vol_ma_20_1d_aligned[i] and
              atr_14_1d_aligned[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0