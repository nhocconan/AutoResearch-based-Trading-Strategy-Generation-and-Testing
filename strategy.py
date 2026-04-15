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
    
    # Calculate daily ATR(14) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(20) for trend filter
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate daily RSI(14) for momentum filter
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 4h Donchian(20) for breakout signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donchian_high = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    signals = np.zeros(n)
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when daily ATR is elevated (> 0.4% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.004 * close[i]
        
        # Long conditions:
        # 1. Price breaks above 4h Donchian high (breakout)
        # 2. Price above daily EMA20 (bullish bias)
        # 3. Daily RSI between 40 and 60 (neutral momentum, avoids extremes)
        # 4. Volatility filter
        if (close[i] > donchian_high_aligned[i] and
            close[i] > ema_20_1d_aligned[i] and
            40 <= rsi_14_1d_aligned[i] <= 60 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 4h Donchian low (breakdown)
        # 2. Price below daily EMA20 (bearish bias)
        # 3. Daily RSI between 40 and 60 (neutral momentum, avoids extremes)
        # 4. Volatility filter
        elif (close[i] < donchian_low_aligned[i] and
              close[i] < ema_20_1d_aligned[i] and
              40 <= rsi_14_1d_aligned[i] <= 60 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA20_RSI14_VolFilter_v1"
timeframe = "4h"
leverage = 1.0