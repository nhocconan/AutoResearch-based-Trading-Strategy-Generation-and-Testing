#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Exponential Moving Average crossover (20/50) with 1d RSI filter and volume confirmation
# Long when 12h EMA20 crosses above EMA50 AND 1d RSI < 70 (avoid overbought) AND volume > 1.2x 20-period average
# Short when 12h EMA20 crosses below EMA50 AND 1d RSI > 30 (avoid oversold) AND volume > 1.2x 20-period average
# ATR trailing stop (2.0x ATR) to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 12h timeframe
# EMA crossover captures trend changes, RSI filter prevents extremes, volume confirmation adds conviction

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d RSI (14-period) as trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === 12h EMA20 and EMA50 ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 12h Volume Spike Confirmation (20-period average) ===
    vol_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # === 12h ATR for trailing stop (14-period) ===
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        rsi_val = rsi_1d_aligned[i]
        ema_20_val = ema_20_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.2  # 1.2x average volume for spike
        atr_val = atr_aligned[i]
        
        # EMA crossover signals
        ema_bullish = ema_20_val > ema_50_val
        ema_bearish = ema_20_val < ema_50_val
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.0*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.0*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: EMA20 > EMA50 (bullish crossover) AND RSI < 70 AND volume spike
            if ema_bullish and rsi_val < 70 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: EMA20 < EMA50 (bearish crossover) AND RSI > 30 AND volume spike
            elif ema_bearish and rsi_val > 30 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_EMA20_50_1dRSI_VolumeConfirm_ATRTrail"
timeframe = "12h"
leverage = 1.0