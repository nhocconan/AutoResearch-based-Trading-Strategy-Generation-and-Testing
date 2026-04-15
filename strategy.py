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
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_ratio = df_1d['volume'].values / (volume_ratio_1d + 1e-10)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Trend filter: price relative to daily EMA34
        trend_filter = close[i] > ema_34_1d_aligned[i]
        
        # Long conditions:
        # 1. Price above daily EMA34 (bullish bias)
        # 2. Price breaks above daily Donchian(20) high with volume (bullish breakout)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Daily volatility regime filter
        if (trend_filter and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio_aligned[i] > 1.8 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below daily EMA34 (bearish bias)
        # 2. Price breaks below daily Donchian(20) low with volume (bearish breakdown)
        # 3. Volume confirmation: volume > 1.8x average
        # 4. Daily volatility regime filter
        elif (not trend_filter and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio_aligned[i] > 1.8 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Vol_Regime_Donchian20_1dEMA34_Breakout_v1"
timeframe = "1d"
leverage = 1.0