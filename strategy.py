#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily OHLC for ATR calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Calculate True Range and ATR(14) on daily ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Calculate 50-day average ATR for volatility regime filter ===
    atr_ma_50 = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # === Align ATR and ATR MA to 4h timeframe ===
    atr_1d_4h = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_50_4h = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    # === Volatility filter: only trade when current ATR > 50-day average ATR ===
    vol_regime = atr_1d_4h > atr_ma_50_4h
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_1d_4h[i]) or np.isnan(atr_ma_50_4h[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(vol_regime[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_ok = vol_regime[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC: Exit when volatility regime changes or volatility drops ===
        if position == 1:  # Long position
            # Exit when volatility regime turns off or ATR drops significantly
            if not vol_ok or (i > 0 and atr_1d_4h[i] < atr_1d_4h[i-1] * 0.8):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when volatility regime turns off or ATR drops significantly
            if not vol_ok or (i > 0 and atr_1d_4h[i] < atr_1d_4h[i-1] * 0.8):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price closes above prior close with volume spike and volatility expansion
            if price > close[i-1] and vol_spike and vol_ok:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price closes below prior close with volume spike and volatility expansion
            elif price < close[i-1] and vol_spike and vol_ok:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Volatility_Expansion_Volume_Spike"
timeframe = "4h"
leverage = 1.0