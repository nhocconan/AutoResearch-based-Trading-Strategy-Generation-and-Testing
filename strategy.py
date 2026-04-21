#!/usr/bin/env python3
"""
Hypothesis: 1h momentum strategy using 4h RSI divergence and volume confirmation.
In both bull and bear markets, momentum shifts often precede price reversals.
RSI divergence on 4h (price makes new high/low but RSI does not) signals weakening momentum.
Combined with 1h volume spike and price action confirmation for precise entry.
Uses 4h for signal direction (lower frequency) and 1h for entry timing to reduce overtrading.
Target: 15-35 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for RSI divergence
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h RSI(14) for divergence detection
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Bearish divergence: price makes higher high, RSI makes lower high
    # Bullish divergence: price makes lower low, RSI makes higher low
    lookback = 10
    bearish_div = np.zeros(len(rsi_4h), dtype=bool)
    bullish_div = np.zeros(len(rsi_4h), dtype=bool)
    
    for i in range(lookback, len(rsi_4h)):
        # Bearish divergence
        if (close_4h[i] == np.max(close_4h[i-lookback:i+1]) and 
            rsi_4h[i] < np.max(rsi_4h[i-lookback:i])):
            bearish_div[i] = True
        # Bullish divergence
        if (close_4h[i] == np.min(close_4h[i-lookback:i+1]) and 
            rsi_4h[i] > np.min(rsi_4h[i-lookback:i])):
            bullish_div[i] = True
    
    bearish_div_aligned = align_htf_to_ltf(prices, df_4h, bearish_div.astype(float))
    bullish_div_aligned = align_htf_to_ltf(prices, df_4h, bullish_div.astype(float))
    
    # Volume confirmation: 1h volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(bearish_div_aligned[i]) or np.isnan(bullish_div_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.8  # Volume spike filter
        
        if position == 0:
            # Enter long: bullish divergence on 4h + volume spike + price above VWAP(20)
            vwap_20 = (pd.Series(prices['close'].values * prices['volume'].values).rolling(20, min_periods=20).sum() / 
                      pd.Series(prices['volume'].values).rolling(20, min_periods=20).sum()).values[i]
            
            if (bullish_div_aligned[i] > 0.5 and 
                vol_ratio_val > vol_threshold and 
                price_close > vwap_20):
                signals[i] = 0.20
                position = 1
            # Enter short: bearish divergence on 4h + volume spike + price below VWAP(20)
            elif (bearish_div_aligned[i] > 0.5 and 
                  vol_ratio_val > vol_threshold and 
                  price_close < vwap_20):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or divergence fails
            # Approximate 1h RSI for exit signal
            if i >= 14:
                rsi_1h = 100 - (100 / (1 + (np.where(avg_loss != 0, avg_gain / avg_loss, 100))))  # Simplified
                # Actually compute properly for 1h
                delta_1h = np.diff(prices['close'].values[:i+1], prepend=prices['close'].values[0])
                gain_1h = np.where(delta_1h > 0, delta_1h, 0)
                loss_1h = np.where(delta_1h < 0, -delta_1h, 0)
                avg_gain_1h = np.zeros_like(gain_1h)
                avg_loss_1h = np.zeros_like(loss_1h)
                if len(gain_1h) > 13:
                    avg_gain_1h[13] = np.mean(gain_1h[1:14])
                    avg_loss_1h[13] = np.mean(loss_1h[1:14])
                    for j in range(14, len(gain_1h)):
                        avg_gain_1h[j] = (avg_gain_1h[j-1] * 13 + gain_1h[j]) / 14
                        avg_loss_1h[j] = (avg_loss_1h[j-1] * 13 + loss_1h[j]) / 14
                    rs_1h = np.where(avg_loss_1h != 0, avg_gain_1h / avg_loss_1h, 100)
                    rsi_1h_val = 100 - (100 / (1 + rs_1h[-1]))
                else:
                    rsi_1h_val = 50
                
                if position == 1 and (rsi_1h_val < 40 or bullish_div_aligned[i] < 0.5):
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and (rsi_1h_val > 60 or bearish_div_aligned[i] < 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    # Hold position
                    signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSIDivergence_Volume_Spike_VWAP"
timeframe = "1h"
leverage = 1.0