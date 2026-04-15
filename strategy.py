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
    
    # Calculate daily EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(df_1d['high'].values - df_1d['low'].values)
    tr2 = pd.Series(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)))
    tr3 = pd.Series(np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    vol_ratio = df_1d['volume'].values / vol_sma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume > 1.2x 20-period average
        vol_filter = vol_ratio_aligned[i] > 1.2
        
        # Long conditions:
        # 1. Price above daily EMA21 (bullish bias)
        # 2. Low volatility environment (ATR below 20-period average)
        # 3. Volume confirmation
        if (close[i] > ema_21_1d_aligned[i] and 
            atr_14_1d_aligned[i] < np.nanmedian(atr_14_1d_aligned[max(0, i-50):i+1]) and 
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA21 (bearish bias)
        # 2. Low volatility environment
        # 3. Volume confirmation
        elif (close[i] < ema_21_1d_aligned[i] and 
              atr_14_1d_aligned[i] < np.nanmedian(atr_14_1d_aligned[max(0, i-50):i+1]) and 
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA21_ATR_VolFilter_v1"
timeframe = "4h"
leverage = 1.0