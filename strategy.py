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
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h ATR(14) for volatility regime filter
    tr1_4h = df_4h['high'] - df_4h['low']
    tr2_4h = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3_4h = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d Donchian(20) channels
    donchian_high_20_1d = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1d = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20_1d)
    donchian_low_20_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20_1d)
    
    # Calculate 1h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_4h_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(donchian_high_20_1d_aligned[i]) or np.isnan(donchian_low_20_1d_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 4h ATR is elevated (> 0.3% of price)
        vol_regime = atr_14_4h_aligned[i] > 0.003 * close[i]
        
        # Trend filter: price relative to 4h EMA50
        trend_filter = close[i] > ema_50_4h_aligned[i]
        
        # Long conditions:
        # 1. Price above 4h EMA50 (bullish bias)
        # 2. Price breaks above 1d Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Volatility regime filter
        if (trend_filter and
            close[i] > donchian_high_20_1d_aligned[i] and
            volume_ratio[i] > 2.0 and
            vol_regime):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. Price below 4h EMA50 (bearish bias)
        # 2. Price breaks below 1d Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Volatility regime filter
        elif (not trend_filter and
              close[i] < donchian_low_20_1d_aligned[i] and
              volume_ratio[i] > 2.0 and
              vol_regime):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Vol_Regime_Donchian20_4hEMA50_Breakout_v1"
timeframe = "1h"
leverage = 1.0