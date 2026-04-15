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
    
    # Get 1d HTF data once before loop (primary HTF for trend and pivots)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Williams %R (14-period) for mean reversion
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 4h ATR(14) for volatility filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([[close_4h[0]], close_4h[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    volume_ratio_4h = vol_4h / (vol_ma_20_4h + 1e-10)
    volume_ratio_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ratio_4h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(atr_14_4h_aligned[i]) or np.isnan(volume_ratio_4h_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 1d Williams %R oversold (< -80) - mean reversion long
        # 2. Price above 1d EMA(50) - bullish trend bias
        # 3. 4h volatility filter: ATR > 0.5% of price (avoid extremely low volatility)
        # 4. 4h volume confirmation: volume > 1.3x average
        if (williams_r_aligned[i] < -80 and
            close[i] > ema_50_1d_aligned[i] and
            atr_14_4h_aligned[i] > 0.005 * close[i] and
            volume_ratio_4h_aligned[i] > 1.3):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 1d Williams %R overbought (> -20) - mean reversion short
        # 2. Price below 1d EMA(50) - bearish trend bias
        # 3. 4h volatility filter: ATR > 0.5% of price
        # 4. 4h volume confirmation: volume > 1.3x average
        elif (williams_r_aligned[i] > -20 and
              close[i] < ema_50_1d_aligned[i] and
              atr_14_4h_aligned[i] > 0.005 * close[i] and
              volume_ratio_4h_aligned[i] > 1.3):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_1d_WilliamsR_EMA50_Volume_ATR_Filter"
timeframe = "4h"
leverage = 1.0