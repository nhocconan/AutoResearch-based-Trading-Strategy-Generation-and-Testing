#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data once for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume / np.where(vol_ma_30 == 0, 1, vol_ma_30) > 1.5
    
    # 6h ATR for stop-loss calculation
    tr_6h = np.maximum(np.abs(high - low), np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_6h[0] = high[0] - low[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_i = high[i]
        low_i = low[i]
        atr_1d_val = atr_1d_aligned[i]
        vol_ok = vol_filter[i]
        atr_6h_val = atr_6h[i]
        
        # Volatility filter: only trade when daily ATR > 0
        vol_regime_ok = atr_1d_val > 0
        
        if position == 0:
            # Long: price breaks above 6-period high with volume and volatility
            if i >= 6:
                max_high_6 = np.max(high[i-6:i])
                if high_i > max_high_6 and vol_ok and vol_regime_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # Short: price breaks below 6-period low with volume and volatility
            elif i >= 6:
                min_low_6 = np.min(low[i-6:i])
                if low_i < min_low_6 and vol_ok and vol_regime_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or mean reversion
            stop_price = entry_price - 1.5 * atr_6h_val
            if low_i < stop_price or (high_i < np.max(high[i-3:i]) and i >= 3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or mean reversion
            stop_price = entry_price + 1.5 * atr_6h_val
            if high_i > stop_price or (low_i > np.min(low[i-3:i]) and i >= 3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolatilityBreakout_ATRStop_V1"
timeframe = "6h"
leverage = 1.0