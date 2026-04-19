#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_ConservativeBreakout_V1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h trend filter: EMA50 > EMA200
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_4h = ema50_4h > ema200_4h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # 1d volatility filter: ATR(14) percentile < 70 (avoid extreme volatility)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile over 50 days
    atr_series = pd.Series(atr_14)
    atr_percentile = atr_series.rolling(window=50, min_periods=20).rank(pct=True).values
    vol_ok = atr_percentile < 0.7  # Only trade when volatility is not extreme
    vol_ok_aligned = align_htf_to_ltf(prices, df_1d, vol_ok)
    
    # 1h entry: Donchian breakout (20-period) with volume confirmation
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        if np.isnan(trend_4h_aligned[i]) or np.isnan(vol_ok_aligned[i]) or \
           np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above 20-period high with volume, in uptrend, normal volatility
            if price > high_20[i] and volume_ok and trend_4h_aligned[i] and vol_ok_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low with volume, in downtrend, normal volatility
            elif price < low_20[i] and volume_ok and not trend_4h_aligned[i] and vol_ok_aligned[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price returns below 20-period high
            if price < high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price returns above 20-period low
            if price > low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals