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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d KAMA(10,2,30) - adaptive trend
    close_1d = pd.Series(df_1d['close'])
    # Efficiency Ratio
    change = abs(close_1d.diff(10))
    volatility = close_1d.diff().abs().rolling(10).sum()
    er = change / (volatility + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(len(close_1d))
    kama[0] = close_1d.iloc[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d.iloc[i] - kama[i-1])
    kama = kama.values
    
    # Align 1d KAMA to 1d (no shift needed as it's already aligned)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate 1d RSI(14)
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Align 1d RSI to 1d
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ATR to 1d
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    
    # Choppiness regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
    # We'll use a simplified version: ATR ratio for regime
    atr_ma_50 = pd.Series(atr_aligned).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_aligned / (atr_ma_50 + 1e-10)
    chop_regime = atr_ratio > 0.8  # Higher volatility = trending regime
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price > KAMA (bullish trend)
        # 2. RSI < 40 (oversold mean reversion opportunity in trending market)
        # 3. Volatility regime: ATR ratio > 0.8 (sufficient volatility for follow-through)
        if (close[i] > kama_aligned[i] and
            rsi_aligned[i] < 40 and
            chop_regime[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price < KAMA (bearish trend)
        # 2. RSI > 60 (overbought mean reversion opportunity in trending market)
        # 3. Volatility regime: ATR ratio > 0.8
        elif (close[i] < kama_aligned[i] and
              rsi_aligned[i] > 60 and
              chop_regime[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_KAMA_RSI_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0