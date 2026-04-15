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
    
    # Calculate 4h ATR(14) for volatility regime filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h EMA(20) for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_4h_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volatility regime filter: only trade when 4h ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_4h_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price above 4h EMA20 (bullish trend)
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Daily volatility regime filter (avoid chop)
        # 4. During active session
        if (close[i] > ema_20_4h_aligned[i] and
            volume_ratio[i] > 1.3 and
            vol_regime and
            in_session):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. Price below 4h EMA20 (bearish trend)
        # 2. Volume confirmation: volume > 1.3x average
        # 3. Daily volatility regime filter
        # 4. During active session
        elif (close[i] < ema_20_4h_aligned[i] and
              volume_ratio[i] > 1.3 and
              vol_regime and
              in_session):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Vol_Regime_EMA20_Trend_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0