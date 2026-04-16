#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 14-period RSI with 1-day trend filter and volume spike
# Long when RSI < 30 (oversold) AND price > daily EMA200 AND volume > 1.5x 20-period average volume
# Short when RSI > 70 (overbought) AND price < daily EMA200 AND volume > 1.5x 20-period average volume
# Exit when RSI crosses back to neutral (40-60 range)
# Designed for low trade frequency (target: 75-200 total trades over 4 years) to minimize fee drag on 4h timeframe
# RSI mean reversion works in both bull and bear markets when filtered by higher timeframe trend
# Volume spike adds confirmation to avoid false signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily EMA200 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 4h RSI(14) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initialize first average
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    # Wilder's smoothing
    for i in range(15, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = 50  # Neutral before enough data
    
    # === 4h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi[i]
        ema_val = ema_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.5  # 1.5x average volume for confirmation
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when RSI crosses above 40 (back to neutral)
            if rsi_val >= 40:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when RSI crosses below 60 (back to neutral)
            if rsi_val <= 60:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: RSI < 30 (oversold) AND price > EMA200 AND volume confirmation
            if rsi_val < 30 and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: RSI > 70 (overbought) AND price < EMA200 AND volume confirmation
            elif rsi_val > 70 and price < ema_val and vol_confirm:
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

name = "4h_RSI14_1dEMA200_Volume1.5x"
timeframe = "4h"
leverage = 1.0