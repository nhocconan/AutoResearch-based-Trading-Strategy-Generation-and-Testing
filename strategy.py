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
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d ADX(14) for trend regime filter
    plus_dm = np.where((df_1d['high'] - df_1d['high'].shift(1)) > (df_1d['low'].shift(1) - df_1d['low']), 
                       np.maximum(df_1d['high'] - df_1d['high'].shift(1), 0), 0)
    minus_dm = np.where((df_1d['low'].shift(1) - df_1d['low']) > (df_1d['high'] - df_1d['high'].shift(1)), 
                        np.maximum(df_1d['low'].shift(1) - df_1d['low'], 0), 0)
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h EMA(20) for trend filter
    ema_20_6h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema_20_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.008 * close[i]
        
        # Trend regime filter: only trade when daily ADX > 25 (trending market)
        trend_regime = adx_14_aligned[i] > 25
        
        # Long conditions:
        # 1. Price above 6h EMA20 (bullish bias)
        # 2. Volume confirmation: volume > 2.0x average
        # 3. Volatility and trend regime filters
        if (close[i] > ema_20_6h[i] and
            volume_ratio[i] > 2.0 and
            vol_regime and
            trend_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 6h EMA20 (bearish bias)
        # 2. Volume confirmation: volume > 2.0x average
        # 3. Volatility and trend regime filters
        elif (close[i] < ema_20_6h[i] and
              volume_ratio[i] > 2.0 and
              vol_regime and
              trend_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Vol_Regime_Trend_Regime_EMA20_Breakout_v1"
timeframe = "6h"
leverage = 1.0