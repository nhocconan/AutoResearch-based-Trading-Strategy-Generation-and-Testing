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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian channel (20-period)
    donchian_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        # This avoids low-volatility chop and focuses on momentum/trend days
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price breaks above 1d Donchian high (20-period breakout)
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Daily volatility regime filter (avoid chop)
        if (close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 1.3 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 1d Donchian low (20-period breakout)
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Daily volatility regime filter
        elif (close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 1.3 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_Volume_Regime_v3"
timeframe = "4h"
leverage = 1.0