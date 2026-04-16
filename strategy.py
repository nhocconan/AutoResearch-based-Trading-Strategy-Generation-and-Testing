#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation
# Long when Williams %R(14) crosses above -80 (oversold reversal) with price > 1d EMA34 and volume > 2x 20-period average
# Short when Williams %R(14) crosses below -20 (overbought reversal) with price < 1d EMA34 and volume > 2x 20-period average
# ATR-based trailing stop (2.5x ATR) to manage risk
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag
# Williams %R is effective in ranging/bear markets for mean reversion, EMA34 filter avoids counter-trend trades

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Williams %R (14-period) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                         ((highest_high_14 - close) / (highest_high_14 - lowest_low_14)) * -100,
                         -50)  # neutral when range=0
    
    # === 6h ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        williams_r_val = williams_r[i]
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r_val
        ema_val = ema_34_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 2.0  # 2x average volume
        atr_val = atr[i]
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from highest
            if atr_val > 0 and price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry or lowest_since_entry == 0:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from lowest
            if atr_val > 0 and price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Williams %R reversal) ===
        if position == 1:  # Long position
            # Exit when Williams %R crosses below -50 (momentum loss)
            if williams_r_val < -50 and williams_r_prev >= -50:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R crosses above -50 (momentum loss)
            if williams_r_val > -50 and williams_r_prev <= -50:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Williams %R crosses above -80 (oversold reversal) AND price > EMA34 AND volume confirmation
            if williams_r_val > -80 and williams_r_prev <= -80 and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
                continue
            # Short when: Williams %R crosses below -20 (overbought reversal) AND price < EMA34 AND volume confirmation
            elif williams_r_val < -20 and williams_r_prev >= -20 and price < ema_val and vol_confirm:
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

name = "6h_WilliamsR_1dEMA34_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0