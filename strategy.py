#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Keltner_Trend_Volume_Entry_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR for trend filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d.iloc[0] = tr1.iloc[0]  # first value
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # 4h EMA for trend direction (short-term)
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # 4h Keltner Channel components
    tr_4h1 = high - low
    tr_4h2 = np.abs(high - np.roll(close, 1))
    tr_4h3 = np.abs(low - np.roll(close, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]  # first value
    atr_4h = pd.Series(tr_4h).rolling(window=10, min_periods=10).mean().values
    ema_10 = pd.Series(close).ewm(span=10, min_periods=10, adjust=False).mean().values
    keltner_upper = ema_10 + 1.5 * atr_4h
    keltner_lower = ema_10 - 1.5 * atr_4h
    
    # Volume confirmation: current volume > 1.8x 20-period average (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(atr_1d[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_4h_val = atr_4h[i]
        ema_val = ema_20[i]
        keltner_up = keltner_upper[i]
        keltner_low = keltner_lower[i]
        atr_1d_val = atr_1d[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price above EMA20 and breaks above Keltner upper with volume
            if price > ema_val and price > keltner_up and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA20 and breaks below Keltner lower with volume
            elif price < ema_val and price < keltner_low and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below EMA20 or below Keltner lower
            if price < ema_val or price < keltner_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above EMA20 or above Keltner upper
            if price > ema_val or price > keltner_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals