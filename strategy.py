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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h Donchian(20) channels
    donchian_high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ratio_12h = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    volume_ratio = volume / (volume_ratio_12h + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 12h ATR is elevated (> 0.7% of price)
        vol_regime = atr_14_12h_aligned[i] > 0.007 * close[i]
        
        # Trend filter: price relative to 12h EMA34
        trend_filter = close[i] > ema_34_12h_aligned[i]
        
        # Long conditions:
        # 1. Price above 12h EMA34 (bullish bias)
        # 2. Price breaks above 12h Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. 12h volatility regime filter
        if (trend_filter and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 2.0 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA34 (bearish bias)
        # 2. Price breaks below 12h Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. 12h volatility regime filter
        elif (not trend_filter and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 2.0 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_EMA34_Donchian20_VolumeBreakout_v1"
timeframe = "4h"
leverage = 1.0