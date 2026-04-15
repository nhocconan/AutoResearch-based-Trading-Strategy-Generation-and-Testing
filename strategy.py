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
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian(20) channels for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    donchian_high_20_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20_4h)
    donchian_low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_20_4h_aligned[i]) or np.isnan(donchian_low_20_4h_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.4% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.004 * close[i]
        
        # Trend filter: price relative to daily EMA50
        trend_filter = close[i] > ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x 4h average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_4h_aligned[i]
        
        # Long conditions:
        # 1. Price above daily EMA50 (bullish bias)
        # 2. Price breaks above 4h Donchian(20) high with volume confirmation
        # 3. Daily volatility regime filter
        if (trend_filter and
            close[i] > donchian_high_20_4h_aligned[i] and
            volume_confirmed and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA50 (bearish bias)
        # 2. Price breaks below 4h Donchian(20) low with volume confirmation
        # 3. Daily volatility regime filter
        elif (not trend_filter and
              close[i] < donchian_low_20_4h_aligned[i] and
              volume_confirmed and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyEMA50_Vol_Regime_Donchian20_4h_Breakout_v1"
timeframe = "4h"
leverage = 1.0