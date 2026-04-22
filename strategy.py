#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar: no previous close
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 4h timeframe (primary timeframe)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter (20-period average on 4h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(atr_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        ema34 = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price pulls back to EMA34 in uptrend with volume
            if price > ema34 and abs(price - ema34) < 0.5 * atr and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to EMA34 in downtrend with volume
            elif price < ema34 and abs(price - ema34) < 0.5 * atr and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price moves 1*ATR away from EMA34 in opposite direction
            if position == 1 and price < ema34 - atr:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > ema34 + atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_EMA34_Pullback_ATR_Volume_Filter"
timeframe = "4h"
leverage = 1.0