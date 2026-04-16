#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves. 1d EMA50 ensures we trade with the higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) ensures breakouts have institutional participation.
# ATR-based trailing stop (2.5x ATR) manages risk and allows trends to run.
# This combination works in both bull and bear markets by only taking breakouts in the direction of the 1d trend.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA50 trend filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === 1d EMA50 ===
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 4h Donchian(20) channels ===
    # We'll calculate Donchian on 4h data then align to primary timeframe
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper channel: 20-period high
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower channel: 20-period low
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to primary timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR for trailing stop (using 14-period) ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    # Set first period TR to high-low (no previous close)
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and extreme price for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    extreme_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50 = ema_50_aligned[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Update extreme price for trailing stop
        if position == 1:  # Long position
            if price > extreme_price:
                extreme_price = price
        elif position == -1:  # Short position
            if extreme_price == 0 or price < extreme_price:
                extreme_price = price
        else:  # Flat position
            extreme_price = 0.0
        
        # === TRAILING STOP LOGIC ===
        if position == 1 and atr_val > 0:  # Long position
            # Exit if price drops 2.5*ATR from extreme price
            if price < extreme_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1 and atr_val > 0:  # Short position
            # Exit if price rises 2.5*ATR from extreme price
            if price > extreme_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = volume[i] > vol_ma * 1.5  # 1.5x average volume
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: price breaks above Donchian upper channel AND above 1d EMA50 AND volume confirmation
            if price > donch_high and price > ema_50 and vol_confirm:
                signals[i] = 0.25
                position = 1
                extreme_price = price
                continue
            # Short when: price breaks below Donchian lower channel AND below 1d EMA50 AND volume confirmation
            elif price < donch_low and price < ema_50 and vol_confirm:
                signals[i] = -0.25
                position = -1
                extreme_price = price
                continue
        
        # === EXIT LOGIC (Donchian opposite channel touch) ===
        elif position == 1:  # Long position
            # Exit when price touches or goes below Donchian lower channel
            if price <= donch_low:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price touches or goes above Donchian upper channel
            if price >= donch_high:
                signals[i] = 0.0
                position = 0
                extreme_price = 0.0
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1dEMA50_VolumeConfirm_ATRTrail"
timeframe = "4h"
leverage = 1.0