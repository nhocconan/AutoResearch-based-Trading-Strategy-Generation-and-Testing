#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA crossover with 12h volume confirmation and 1d RSI mean reversion filter
# Long when 6h EMA(9) crosses above EMA(21) AND volume > 1.5x 12h average volume AND 1d RSI(14) < 40
# Short when 6h EMA(9) crosses below EMA(21) AND volume > 1.5x 12h average volume AND 1d RSI(14) > 60
# ATR trailing stop (2.0x ATR) to manage risk
# EMA crossover captures momentum, volume confirms conviction, RSI filters for mean reversion opportunities
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h EMA crossover (9 and 21) ===
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 12h Volume Confirmation (average volume) ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values  # 12 days average
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # === 1d RSI mean reversion filter (14-period) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema9[i]) or 
            np.isnan(ema21[i]) or
            np.isnan(vol_ma_12h_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_12h_aligned[i]
        rsi_val = rsi_aligned[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.5x 12h average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # EMA crossover signals
        ema9_prev = ema9[i-1] if i > 0 else ema9[i]
        ema21_prev = ema21[i-1] if i > 0 else ema21[i]
        ema_cross_up = ema9[i] > ema21[i] and ema9_prev <= ema21_prev
        ema_cross_down = ema9[i] < ema21[i] and ema9_prev >= ema21_prev
        
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
            # Long when: EMA bullish crossover AND volume confirmation AND RSI < 40 (oversold)
            if ema_cross_up and vol_confirm and rsi_val < 40:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: EMA bearish crossover AND volume confirmation AND RSI > 60 (overbought)
            elif ema_cross_down and vol_confirm and rsi_val > 60:
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

name = "6h_EMA9_21_12hVolume1.5x_1dRSI40_60_ATRTrail_2.0x"
timeframe = "6h"
leverage = 1.0