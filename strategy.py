#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Pivot_R3S3_Reversal_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d: Calculate Daily Pivot Points (Standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # R3 = P + 2*(H - L), S3 = P - 2*(H - L) [Standard formula]
    r3_1d = pivot_1d + 2 * range_1d
    s3_1d = pivot_1d - 2 * range_1d
    
    # Align Pivot levels
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 6h: Indicators ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI(14) for overbought/oversold
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
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
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        current_rsi = rsi[i]
        current_atr = atr[i]
        current_close = close[i]
        current_volume = volume[i]
        
        # Skip if any value is NaN
        if (np.isnan(r3) or np.isnan(s3) or np.isnan(current_rsi) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.3x 20-period 6h average volume
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            vol_condition = current_volume > 1.3 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long reversal: price touches/bounces off S3 with RSI oversold + volume
            if current_close <= s3 and current_rsi < 30 and vol_condition:
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short reversal: price touches/rejects at R3 with RSI overbought + volume
            elif current_close >= r3 and current_rsi > 70 and vol_condition:
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit: price reaches R3 OR stop loss OR RSI overbought
            if current_close >= r3 or current_close < entry_price - 2.0 * current_atr or current_rsi > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S3 OR stop loss OR RSI oversold
            if current_close <= s3 or current_close > entry_price + 2.0 * current_atr or current_rsi < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals