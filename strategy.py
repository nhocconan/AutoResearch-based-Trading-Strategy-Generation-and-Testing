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
    
    # Calculate 1d RSI(14) for momentum filter
    close_1d = pd.Series(df_1d['close'].values)
    delta = close_1d.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(50).values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d_shift = close_1d.shift(1)
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d_shift)
    tr3 = abs(low_1d - close_1d_shift)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    atr_14_values = atr_14.fillna(0).values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_values)
    
    # Calculate 6h ATR(14) for position sizing normalization
    tr_6h = pd.Series(np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.concatenate([[np.nan], close[:-1]])),
            np.abs(low - np.concatenate([[np.nan], close[:-1]]))
        )
    ))
    atr_6h = tr_6h.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(atr_6h[i]) or atr_6h[i] == 0):
            signals[i] = 0.0
            continue
        
        # Normalize 6h ATR by 1d ATR for volatility regime filter
        atr_ratio = atr_6h[i] / atr_14_aligned[i]
        
        # Long conditions:
        # 1. 1d RSI < 30 (oversold on higher timeframe)
        # 2. Current 6h volatility is low relative to daily (mean reversion setup)
        # 3. Price below 6h VWAP (mean reentry opportunity)
        if (rsi_14_aligned[i] < 30 and 
            atr_ratio < 0.8 and 
            close[i] < np.mean([high[i], low[i], close[i]]) * (volume[i] > 0)):  # Simplified VWAP proxy
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d RSI > 70 (overbought on higher timeframe)
        # 2. Current 6h volatility is low relative to daily (mean reversion setup)
        # 3. Price above 6h VWAP (mean reentry opportunity)
        elif (rsi_14_aligned[i] > 70 and 
              atr_ratio < 0.8 and 
              close[i] > np.mean([high[i], low[i], close[i]]) * (volume[i] > 0)):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_RSI14_MeanReversion_VolFilter_v1"
timeframe = "6h"
leverage = 1.0