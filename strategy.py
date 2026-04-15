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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    hh_12h = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    williams_r_12h = ((hh_12h - df_12h['close'].values) / (hh_12h - ll_12h + 1e-10)) * -100
    
    # Align HTF Williams %R to 6h timeframe
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h Williams %R oversold (< -80) with 6h price above 6h EMA(20) → long
        # 2. 12h Williams %R overbought (> -20) with 6h price below 6h EMA(20) → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Calculate 6h EMA(20)
        if i >= 20:
            ema_20 = pd.Series(close[:i+1]).ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1]
        else:
            signals[i] = 0.0
            continue
        
        # Long conditions: 12h oversold + 6h price above EMA(20)
        if (williams_r_12h_aligned[i] < -80 and            # 12h Williams %R oversold
            close[i] > ema_20 and                          # 6h price above 6h EMA(20)
            volume_ratio[i] > 1.3 and                      # Volume confirmation
            atr_14[i] > 0.005 * close[i]):                 # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 12h overbought + 6h price below EMA(20)
        elif (williams_r_12h_aligned[i] > -20 and          # 12h Williams %R overbought
              close[i] < ema_20 and                        # 6h price below 6h EMA(20)
              volume_ratio[i] > 1.3 and                    # Volume confirmation
              atr_14[i] > 0.005 * close[i]):               # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_12h_Oversold_Overbought_EMA20_Filter"
timeframe = "6h"
leverage = 1.0