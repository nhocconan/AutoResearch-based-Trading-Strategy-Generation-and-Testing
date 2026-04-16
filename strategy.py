#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h volume spike and 1d trend filter
# Long when price breaks above Camarilla R3 on 1h with 4h volume > 2x 20-period average and price > 1d EMA50
# Short when price breaks below Camarilla S3 on 1h with 4h volume > 2x 20-period average and price < 1d EMA50
# ATR-based trailing stop (1.5x ATR) to manage risk
# Uses 4h for volume confirmation and 1d for trend direction, 1h for precise entry timing
# Target: 60-120 total trades over 4 years to minimize fee drag while capturing intraday moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Camarilla Pivots (R3, S3) ===
    # Calculate from previous 1h bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hL = prev_high - prev_low
    camarilla_r3 = pivot + range_hL * 1.1 / 4.0
    camarilla_s3 = pivot - range_hL * 1.1 / 4.0
    
    # === 4h Volume Confirmation (20-period average) ===
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ma_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20)
    
    # === 1d EMA50 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 1h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
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
        if (np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(ema_aligned[i]) or
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r3_val = camarilla_r3[i]
        s3_val = camarilla_s3[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 2.0  # 2x average volume
        ema_val = ema_aligned[i]
        atr_val = atr_1h[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 1.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 1.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (trend filter reversal) ===
        if position == 1:  # Long position
            # Exit when price crosses below 1d EMA50
            if price < ema_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses above 1d EMA50
            if price > ema_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Camarilla R3 AND 4h volume confirmation AND price > EMA50
            if price > r3_val and vol_confirm and price > ema_val:
                signals[i] = 0.20
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: price breaks below Camarilla S3 AND 4h volume confirmation AND price < EMA50
            elif price < s3_val and vol_confirm and price < ema_val:
                signals[i] = -0.20
                position = -1
                entry_price = price
                lowest_since_entry = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R3S3_4hVolumeSpike_1dEMA50_ATRTrail"
timeframe = "1h"
leverage = 1.0