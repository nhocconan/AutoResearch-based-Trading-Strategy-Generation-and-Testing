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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly ATR(10) for volatility regime filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_1w = pd.Series(tr_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Calculate weekly Donchian channel (20-period)
    highest_high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1w, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_20)
    
    # Calculate 1d ATR(14) for volatility entry filter
    tr1_1d = high - low
    tr2_1d = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_1d = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_10_1w_aligned[i]) or np.isnan(highest_high_20_aligned[i]) or 
            np.isnan(lowest_low_20_aligned[i]) or np.isnan(atr_14_1d[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when weekly ATR is elevated (> 0.6% of price)
        # This avoids low-volatility chop and focuses on momentum/trend weeks
        vol_regime = atr_10_1w_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian high (bullish breakout)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Weekly volatility regime filter (avoid chop)
        if (close[i] > highest_high_20_aligned[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian low (bearish breakout)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. Weekly volatility regime filter
        elif (close[i] < lowest_low_20_aligned[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Donchian20_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0