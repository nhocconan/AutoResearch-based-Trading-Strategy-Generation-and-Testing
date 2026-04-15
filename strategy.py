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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for long-term trend
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1w ATR(14) for volatility regime filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate 1w Donchian(20) channels
    donchian_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when weekly ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1w_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price above weekly EMA34 (bullish long-term bias)
        # 2. Price breaks above weekly Donchian(20) high with volume confirmation
        # 3. Volume spike: volume > 2.0x average daily volume
        if (close[i] > ema_34_1w_aligned[i] and
            close[i] > donchian_high_20_aligned[i] and
            volume[i] > 2.0 * np.mean(volume[max(0, i-20):i+1]) and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA34 (bearish long-term bias)
        # 2. Price breaks below weekly Donchian(20) low with volume confirmation
        # 3. Volume spike: volume > 2.0x average daily volume
        elif (close[i] < ema_34_1w_aligned[i] and
              close[i] < donchian_low_20_aligned[i] and
              volume[i] > 2.0 * np.mean(volume[max(0, i-20):i+1]) and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Vol_Regime_Donchian20_1wEMA34_Breakout_v1"
timeframe = "1d"
leverage = 1.0