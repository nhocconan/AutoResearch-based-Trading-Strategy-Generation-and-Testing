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
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d ATR(14) for volatility filter ===
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d ATR(50) for long-term volatility ===
    atr_50_1d = pd.Series(tr_1d).rolling(window=50, min_periods=50).mean().values
    atr_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_50_1d)
    
    # === 6h ATR(14) for position sizing ===
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === 6h Volume ratio (current vs 20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_6h / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_6h, vol_ratio)
    
    # === 6h RSI(14) for momentum ===
    delta = np.diff(close_6h, prepend=close_6h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_50_1d_aligned[i]) or np.isnan(vol_ratio_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        atr_6h_val = atr_6h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_50_1d_val = atr_50_1d_aligned[i]
        vol_ratio_val = vol_ratio_aligned[i]
        rsi_val = rsi[i]
        
        # Volatility regime: short-term ATR vs long-term ATR
        vol_regime = atr_6h_val / atr_50_1d_val if atr_50_1d_val > 0 else 1.0
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when volatility drops (range-bound) OR RSI overbought
            if vol_regime < 0.8 or rsi_val > 70:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when volatility drops OR RSI oversold
            if vol_regime < 0.8 or rsi_val < 30:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: High volume + momentum + volatility expansion
            if (vol_ratio_val > 1.5) and (rsi_val > 55) and (vol_regime > 1.2):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: High volume + momentum + volatility expansion
            elif (vol_ratio_val > 1.5) and (rsi_val < 45) and (vol_regime > 1.2):
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

name = "6h_VolMom_VolRegime_Expansion"
timeframe = "6h"
leverage = 1.0