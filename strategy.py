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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend direction
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1d HTF data for session filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1_1d = df_1d['high'] - df_1d['low']
    tr2_1d = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3_1d = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volatility regime filter: only trade when 4h ATR is elevated (> 0.8% of price)
        vol_regime_4h = atr_14_4h_aligned[i] > 0.008 * close[i]
        
        # Additional volatility filter: avoid extremely high volatility days (> 4% ATR)
        vol_not_extreme = atr_14_4h_aligned[i] <= 0.04 * close[i]
        
        # Daily volatility filter: only trade when daily ATR is elevated (> 0.6% of price)
        vol_regime_1d = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price above 4h EMA20 (bullish bias)
        # 2. 4h volatility regime (avoid chop)
        # 3. Not extreme volatility
        # 4. Daily volatility regime
        # 5. In trading session
        if (close[i] > ema_20_4h_aligned[i] and
            vol_regime_4h and
            vol_not_extreme and
            vol_regime_1d and
            in_session):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. Price below 4h EMA20 (bearish bias)
        # 2. 4h volatility regime (avoid chop)
        # 3. Not extreme volatility
        # 4. Daily volatility regime
        # 5. In trading session
        elif (close[i] < ema_20_4h_aligned[i] and
              vol_regime_4h and
              vol_not_extreme and
              vol_regime_1d and
              in_session):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_EMA20_4h_Vol_Regime_Session_Filter"
timeframe = "1h"
leverage = 1.0