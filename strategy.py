#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR for volatility regime filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Load 4h data
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h ATR for position sizing and stop
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_low = high_4h - low_4h
    high_close = np.abs(high_4h - np.roll(close_4h, 1))
    low_close = np.abs(low_4h - np.roll(close_4h, 1))
    high_low[0] = high_4h[0] - low_4h[0]
    high_close[0] = np.abs(high_4h[0] - close_4h[0])
    low_close[0] = np.abs(low_4h[0] - close_4h[0])
    tr_4h = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_4h[0] = high_low[0]
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (20-period)
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_10[i]) or np.isnan(atr_14_4h[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vol = volume_4h[i]
        
        # Volatility filter: trade only when volatility is elevated (avoid chop)
        vol_filter = atr_10[i] > np.nanmedian(atr_10[max(0, i-50):i+1])
        
        if position == 0 and vol_filter:
            # Long: price > daily ATR-based upper band with volume spike
            upper_band = close_1d[i] + 0.5 * atr_10[i]
            if price > upper_band and vol > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < daily ATR-based lower band with volume spike
            elif price < close_1d[i] - 0.5 * atr_10[i] and vol > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below daily close or ATR stop
            if price < close_1d[i] or price < df_4h['high'].values[i] - 1.5 * atr_14_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above daily close or ATR stop
            if price > close_1d[i] or price > df_4h['low'].values[i] + 1.5 * atr_14_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_ATRBreakout_VolumeSpike"
timeframe = "4h"
leverage = 1.0