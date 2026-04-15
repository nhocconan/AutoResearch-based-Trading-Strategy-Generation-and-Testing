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
    
    # Get 4h HTF data once before loop (trend filter)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop (volatility regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h ATR(14) for volatility entry filter
    tr1_1h = high - low
    tr2_1h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_1h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_1h = np.maximum(tr1_1h, np.maximum(tr2_1h, tr3_1h))
    atr_14_1h = pd.Series(tr_1h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(atr_14_1h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours (08-20 UTC)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price above 4h EMA50 (bullish trend)
        # 2. 1h ATR > 0.3% of price (sufficient volatility for move)
        # 3. Volume confirmation: volume > 1.2x average
        # 4. Daily volatility regime filter (avoid chop)
        # 5. Session filter
        if (close[i] > ema_50_4h_aligned[i] and
            atr_14_1h[i] > 0.003 * close[i] and
            volume_ratio[i] > 1.2 and
            vol_regime and
            in_session):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. Price below 4h EMA50 (bearish trend)
        # 2. 1h ATR > 0.3% of price
        # 3. Volume confirmation: volume > 1.2x average
        # 4. Daily volatility regime filter
        # 5. Session filter
        elif (close[i] < ema_50_4h_aligned[i] and
              atr_14_1h[i] > 0.003 * close[i] and
              volume_ratio[i] > 1.2 and
              vol_regime and
              in_session):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA50_1d_Vol_Regime_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0