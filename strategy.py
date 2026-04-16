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
    
    # === 6h data (primary) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1w data (HTF for regime filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === 6h indicators ===
    # EMA(50) on 6h
    close_6h_series = pd.Series(close_6h)
    ema_50_6h = close_6h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ATR(14) on 6h for volatility
    tr1 = np.abs(high_6h - low_6h)
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d indicators ===
    # EMA(200) on 1d for long-term trend
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # === 1w indicators ===
    # EMA(50) on 1w for regime filter
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # === Align HTF indicators to 6h timeframe ===
    ema_50_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_50_6h)
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h_aligned[i]) or np.isnan(atr_6h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema_50_6h_val = ema_50_6h_aligned[i]
        atr_6h_val = atr_6h_aligned[i]
        ema_200_1d_val = ema_200_1d_aligned[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below EMA50(6h) OR ATR exceeds 2x median (volatility filter)
            if (price < ema_50_6h_val) or (atr_6h_val > 2.0 * np.median(atr_6h_aligned[max(0, i-50):i+1])):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA50(6h) OR ATR exceeds 2x median
            if (price > ema_50_6h_val) or (atr_6h_val > 2.0 * np.median(atr_6h_aligned[max(0, i-50):i+1])):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Determine market regime using 1w EMA50
            bull_regime = ema_50_1w_val > np.mean(ema_50_1w_aligned[max(0, i-50):i+1])
            bear_regime = ema_50_1w_val < np.mean(ema_50_1w_aligned[max(0, i-50):i+1])
            
            # LONG: Price above EMA50(6h) AND above EMA200(1d) AND in bull regime
            if (price > ema_50_6h_val) and (price > ema_200_1d_val) and bull_regime:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price below EMA50(6h) AND below EMA200(1d) AND in bear regime
            elif (price < ema_50_6h_val) and (price < ema_200_1d_val) and bear_regime:
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

name = "6h_EMA50_EMA200_1w_Regime_Filter"
timeframe = "6h"
leverage = 1.0