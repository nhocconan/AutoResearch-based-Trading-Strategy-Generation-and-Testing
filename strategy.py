#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ElderRay_Power_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d: Elder Ray Power Index ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 of close for 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Smooth the power values with EMA13
    bull_power_smooth = pd.Series(bull_power_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power_smooth = pd.Series(bear_power_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align to 4h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    # === 4h: Indicators ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stop loss
    high = prices['high'].values
    low = prices['low'].values
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
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        current_vol_ma = vol_ma[i]
        current_volume = volume[i]
        current_close = close[i]
        current_atr = atr[i]
        
        # Skip if any value is NaN
        if (np.isnan(bull_val) or np.isnan(bear_val) or 
            np.isnan(current_vol_ma) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8x 20-period average
        vol_condition = current_volume > 1.8 * current_vol_ma
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (strong bullish momentum) + volume
            if bull_val > 0 and bear_val < 0 and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            # Short: Bull Power < 0 AND Bear Power > 0 (strong bearish momentum) + volume
            elif bull_val < 0 and bear_val > 0 and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: Bull Power <= 0 OR Bear Power >= 0 OR stop loss
            if bull_val <= 0 or bear_val >= 0 or current_close < entry_price - 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power >= 0 OR Bear Power <= 0 OR stop loss
            if bull_val >= 0 or bear_val <= 0 or current_close > entry_price + 2.0 * current_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals