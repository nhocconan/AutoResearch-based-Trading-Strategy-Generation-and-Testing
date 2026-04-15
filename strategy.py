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
    
    # Calculate 1d EMA(21) for trend filter
    ema_21_1d = pd.Series(df_1d['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_ratio = volume / (volume_ratio_1d + 1e-10)
    
    # Calculate 4h Donchian(20) channels for entry timing
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donchian_high_20_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20_4h)
    donchian_low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20_4h)
    
    signals = np.zeros(n)
    position = 0  # track current position: 0=flat, 1=long, -1=short
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(donchian_high_20_4h_aligned[i]) or 
            np.isnan(donchian_low_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.5% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Trend filter: price relative to daily EMA21
        trend_filter = close[i] > ema_21_1d_aligned[i]
        
        # Exit conditions: close position if trend reverses or volatility drops
        if position == 1 and (not trend_filter or not vol_regime):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (trend_filter or not vol_regime):
            signals[i] = 0.0
            position = 0
        # Entry conditions: only enter when flat
        elif position == 0:
            # Long conditions:
            # 1. Price above daily EMA21 (bullish bias)
            # 2. Price breaks above 4h Donchian(20) high with volume
            # 3. Volume confirmation: volume > 2.0x average
            # 4. Daily volatility regime filter
            if (trend_filter and
                close[i] > donchian_high_20_4h_aligned[i] and
                volume_ratio[i] > 2.0 and
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Price below daily EMA21 (bearish bias)
            # 2. Price breaks below 4h Donchian(20) low with volume
            # 3. Volume confirmation: volume > 2.0x average
            # 4. Daily volatility regime filter
            elif (not trend_filter and
                  close[i] < donchian_low_20_4h_aligned[i] and
                  volume_ratio[i] > 2.0 and
                  vol_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else -0.25 if position == -1 else 0.0
    
    return signals

name = "4h_Donchian20_1dEMA21_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0