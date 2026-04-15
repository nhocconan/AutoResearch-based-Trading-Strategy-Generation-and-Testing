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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily volume SMA(20) for volume confirmation
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 6h volume > 1.5x daily average volume (scaled to 6h)
        # Approximate: 6h volume should be > 0.5x daily volume (4x6h = 1d)
        vol_filter = volume[i] > 0.5 * vol_sma_20_aligned[i]
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Long breakout: price breaks above daily Donchian high with volume and volatility
        if (close[i] > donchian_high_20_aligned[i] and vol_filter and vol_regime):
            signals[i] = 0.25
            
        # Short breakout: price breaks below daily Donchian low with volume and volatility
        elif (close[i] < donchian_low_20_aligned[i] and vol_filter and vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_Volume_Volatility_v1"
timeframe = "6h"
leverage = 1.0