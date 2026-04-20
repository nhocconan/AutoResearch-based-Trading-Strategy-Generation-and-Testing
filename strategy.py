#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Pivot_R3S3_Breakout_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla Pivot Points (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla R3 and S3 levels
    # R3 = Close + (High - Low) * 1.1 / 4
    # S3 = Close - (High - Low) * 1.1 / 4
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align R3 and S3 to 12h
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 12h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stop loss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(40, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Get values
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        current_high = high[i]
        current_low = low[i]
        current_atr = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3_val) or np.isnan(s3_val) or 
            np.isnan(current_vol_ma) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.5x 20-period average
        vol_condition = current_volume > 2.5 * current_vol_ma
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation
            if current_high > r3_val and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: price breaks below S3 with volume confirmation
            elif current_low < s3_val and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price falls below S3 OR stop loss
            if current_low < s3_val or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R3 OR stop loss
            if current_high > r3_val or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals