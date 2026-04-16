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
    
    # Calculate ATR on 6h
    tr_6h = np.maximum(high_6h - low_6h,
                       np.maximum(np.abs(high_6h - np.roll(close_6h, 1)),
                                  np.abs(low_6h - np.roll(close_6h, 1))))
    tr_6h[0] = high_6h[0] - low_6h[0]
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    # === Daily data (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR on daily
    tr_1d = np.maximum(high_1d - low_1d,
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === Weekly data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR on weekly
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # === Weekly volatility regime: high vol when ATR(1w) > ATR(1d) * 1.5 ===
    vol_regime_high = atr_1w_aligned > (atr_1d_aligned * 1.5)
    
    # === 6h EMA(20) for trend filter ===
    ema_20_6h = pd.Series(close_6h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(ema_20_6h_aligned[i]) or
            np.isnan(vol_regime_high[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        ema_20_val = ema_20_6h_aligned[i]
        atr_6h_val = atr_6h_aligned[i]
        atr_1d_val = atr_1d_aligned[i]
        atr_1w_val = atr_1w_aligned[i]
        vol_regime = vol_regime_high[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below 6h EMA(20) OR volatility regime shifts to high
            if (price < ema_20_val) or vol_regime:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above 6h EMA(20) OR volatility regime shifts to high
            if (price > ema_20_val) or vol_regime:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price above EMA(20) AND low volatility regime (weekly ATR not elevated)
            if (price > ema_20_val) and (not vol_regime):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below EMA(20) AND low volatility regime (weekly ATR not elevated)
            elif (price < ema_20_val) and (not vol_regime):
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

name = "6h_EMA20_VolRegime_Filter"
timeframe = "6h"
leverage = 1.0