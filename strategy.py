#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_R3S3_Breakout_Volume_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    r3_1d = close_1d + range_1d * 1.1 / 2.0
    s3_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 4h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA34 for trend filter
    close_s = pd.Series(close)
    ema34 = close_s.ewm(span=34, min_periods=34, adjust=False).mean().values
    
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
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        current_ema34 = ema34[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3) or np.isnan(s3) or np.isnan(current_ema34) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 20-period 4h average volume ===
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions: break above R3 with volume AND above EMA34 (uptrend)
            if current_close > r3 and vol_condition and current_close > current_ema34:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions: break below S3 with volume AND below EMA34 (downtrend)
            elif current_close < s3 and vol_condition and current_close < current_ema34:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price fails to hold above R3 OR stop loss
            if current_close <= r3 or current_close < entry_price - 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price fails to hold below S3 OR stop loss
            if current_close >= s3 or current_close > entry_price + 2.5 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals