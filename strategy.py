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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian(50) channels
    donchian_high_50 = pd.Series(df_1w['high'].values).rolling(window=50, min_periods=50).max().values
    donchian_low_50 = pd.Series(df_1w['low'].values).rolling(window=50, min_periods=50).min().values
    donchian_high_50_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_50)
    donchian_low_50_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_50)
    
    # Calculate 6h EMA(20) for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h ATR(14) for volatility regime
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_50_aligned[i]) or np.isnan(donchian_low_50_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 6h ATR is elevated (> 0.8% of price)
        vol_regime = atr_14[i] > 0.008 * close[i]
        
        # Trend filter: price relative to 6h EMA20
        trend_filter = close[i] > ema_20[i]
        
        # Long conditions:
        # 1. Price above 6h EMA20 (bullish bias)
        # 2. Price breaks above weekly Donchian(50) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility regime filter
        if (trend_filter and
            close[i] > donchian_high_50_aligned[i] and
            volume_ratio[i] > 1.5 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 6h EMA20 (bearish bias)
        # 2. Price breaks below weekly Donchian(50) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility regime filter
        elif (not trend_filter and
              close[i] < donchian_low_50_aligned[i] and
              volume_ratio[i] > 1.5 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyDonchian50_EMA20_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0