#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeTrend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Donchian(20) channels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period rolling high/low
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donchian_high_20 = align_htf_to_ltf(prices, df_1d, high_max_20)
    donchian_low_20 = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # === 4h: ATR(14) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 4h: 20-period volume moving average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 60  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        upper = donchian_high_20[i]
        lower = donchian_low_20[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(upper) or np.isnan(lower) or np.isnan(current_atr) or np.isnan(vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        vol_condition = current_volume > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper Donchian band with volume
            if current_close > upper and vol_condition:
                signals[i] = 0.30
                position = 1
                entry_price = current_close
            # Short: break below lower Donchian band with volume
            elif current_close < lower and vol_condition:
                signals[i] = -0.30
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: close below lower Donchian band or ATR stop
            if current_close < lower or current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: close above upper Donchian band or ATR stop
            if current_close > upper or current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals