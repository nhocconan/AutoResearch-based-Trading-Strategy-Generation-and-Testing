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
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Donchian(20) channels from prior day's data
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    
    # Donchian upper/lower from prior day's 20-period high/low
    donchian_upper_20 = pd.Series(prior_high).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(prior_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_20)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        # This avoids low-volatility chop and focuses on momentum/trend days
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price breaks above prior day's Donchian upper channel with volume
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Daily volatility regime filter (avoid chop)
        if (close[i] > donchian_upper_aligned[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below prior day's Donchian lower channel with volume
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Daily volatility regime filter
        elif (close[i] < donchian_lower_aligned[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Volume_VolatilityRegime_v1"
timeframe = "12h"
leverage = 1.0