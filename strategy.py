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
    
    # === 12h data (HTF) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate ATR on 12h
    tr_12h = np.maximum(high_12h - low_12h,
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 12h RSI(14) for momentum filter ===
    delta_12h = np.diff(close_12h, prepend=close_12h[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).rolling(window=14, min_periods=14).mean().values
    avg_loss_12h = pd.Series(loss_12h).rolling(window=14, min_periods=14).mean().values
    rs_12h = np.where(avg_loss_12h != 0, avg_gain_12h / avg_loss_12h, 0)
    rsi_14_12h = 100 - (100 / (1 + rs_12h))
    rsi_14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_14_12h)
    
    # === 6h RSI(14) for momentum filter ===
    delta_6h = np.diff(close_6h, prepend=close_6h[0])
    gain_6h = np.where(delta_6h > 0, delta_6h, 0)
    loss_6h = np.where(delta_6h < 0, -delta_6h, 0)
    avg_gain_6h = pd.Series(gain_6h).rolling(window=14, min_periods=14).mean().values
    avg_loss_6h = pd.Series(loss_6h).rolling(window=14, min_periods=14).mean().values
    rs_6h = np.where(avg_loss_6h != 0, avg_gain_6h / avg_loss_6h, 0)
    rsi_14_6h = 100 - (100 / (1 + rs_6h))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_6h_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(rsi_14_12h_aligned[i]) or np.isnan(rsi_14_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close_6h[i]
        rsi_12h_val = rsi_14_12h_aligned[i]
        rsi_6h_val = rsi_14_6h[i]
        atr_6h_val = atr_6h_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when RSI(12h) > 70 (overbought) OR RSI(6h) < 30 (oversold on lower TF)
            if (rsi_12h_val > 70) or (rsi_6h_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI(12h) < 30 (oversold) OR RSI(6h) > 70 (overbought on lower TF)
            if (rsi_12h_val < 30) or (rsi_6h_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: RSI(12h) between 30 and 50 (bullish momentum) AND RSI(6h) > 50 (confirmation)
            if (30 <= rsi_12h_val <= 50) and (rsi_6h_val > 50):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: RSI(12h) between 50 and 70 (bearish momentum) AND RSI(6h) < 50 (confirmation)
            elif (50 <= rsi_12h_val <= 70) and (rsi_6h_val < 50):
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

name = "6h_RSI12h_RSI6h_Momentum_Confluence"
timeframe = "6h"
leverage = 1.0