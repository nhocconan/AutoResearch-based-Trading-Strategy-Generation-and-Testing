#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h HTF data once before loop (primary timeframe is 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend filter
    ema_21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h Donchian(20) channels
    donchian_high_20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    volume_ratio = volume / (vol_ma_20_aligned + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when 4h ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_4h_aligned[i] > 0.003 * close[i]
        
        # Trend filter: price relative to 4h EMA21
        trend_filter = close[i] > ema_21_4h_aligned[i]
        
        # Long conditions:
        # 1. Price above 4h EMA21 (bullish bias)
        # 2. Price breaks above 4h Donchian(20) high with volume confirmation
        # 3. Volume > 1.5x average
        # 4. Volatility filter
        if (trend_filter and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 1.5 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 4h EMA21 (bearish bias)
        # 2. Price breaks below 4h Donchian(20) low with volume confirmation
        # 3. Volume > 1.5x average
        # 4. Volatility filter
        elif (not trend_filter and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 1.5 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA21_Donchian20_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0