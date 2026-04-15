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
    
    # Calculate 1d KAMA (Kaufman Adaptive Moving Average) for trend direction
    # Efficiency Ratio (ER) over 10 periods
    change_1d = np.abs(df_1d['close'].diff(10).values)
    volatility_1d = np.abs(df_1d['close'].diff(1).rolling(window=10, min_periods=10).sum().values)
    er_1d = np.where(volatility_1d > 0, change_1d / volatility_1d, 0)
    # Smoothing constants
    sc_1d = er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1)  # fast=2, slow=30
    sc_1d = np.where(np.isnan(sc_1d), 0, sc_1d)
    sc_1d = sc_1d ** 2  # ER^2 for smoothing
    # Calculate KAMA
    kama_1d = np.full_like(df_1d['close'].values, np.nan, dtype=float)
    kama_1d[9] = df_1d['close'].iloc[9]  # seed
    for i in range(10, len(kama_1d)):
        if not np.isnan(kama_1d[i-1]) and not np.isnan(sc_1d[i]):
            kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (df_1d['close'].iloc[i] - kama_1d[i-1])
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 1d RSI(14) for momentum filter
    delta = df_1d['close'].diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.6% of price)
        # This avoids low-volatility chop and focuses on momentum days
        vol_regime = atr_14_1d_aligned[i] > 0.006 * close[i]
        
        # Long conditions:
        # 1. Price above KAMA (bullish trend)
        # 2. RSI > 50 (bullish momentum)
        # 3. Volatility regime filter
        if (close[i] > kama_1d_aligned[i] and
            rsi_1d_aligned[i] > 50 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below KAMA (bearish trend)
        # 2. RSI < 50 (bearish momentum)
        # 3. Volatility regime filter
        elif (close[i] < kama_1d_aligned[i] and
              rsi_1d_aligned[i] < 50 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_Vol_Regime_v1"
timeframe = "1d"
leverage = 1.0