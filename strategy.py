#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) high/low for price channel
    donch_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # Calculate daily ATR(14) for volatility filter and stop sizing
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
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.2 * 20-day average
        vol_confirm = df_1d['volume'].iloc[i] > 1.2 * vol_sma_20_aligned[i] if i < len(df_1d) else False
        
        # Long conditions:
        # 1. Price breaks above daily Donchian(20) high
        # 2. Volume confirmation
        # 3. ATR filter: avoid extremely low volatility periods
        if (close[i] > donch_high_20_aligned[i] and
            vol_confirm and
            atr_14_1d_aligned[i] > 0.003 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below daily Donchian(20) low
        # 2. Volume confirmation
        # 3. ATR filter: avoid extremely low volatility periods
        elif (close[i] < donch_low_20_aligned[i] and
              vol_confirm and
              atr_14_1d_aligned[i] > 0.003 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_v1"
timeframe = "12h"
leverage = 1.0