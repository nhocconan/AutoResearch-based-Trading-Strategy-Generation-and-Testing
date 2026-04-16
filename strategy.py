#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d EMA34 Trend Filter and Volume Spike Confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Using 1d EMA34 as trend filter: only take long Elder Ray signals when price > EMA34(1d), short when < EMA34(1d)
# Volume confirmation (>1.5x 20-period average) ensures institutional participation
# This combination works in both bull and bear markets by aligning with higher timeframe trend
# Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA34 trend filter (HTF) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === EMA34 on 1d ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === 6h data for Elder Ray calculation ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === Elder Ray Components (6h) ===
    # EMA13 for Elder Ray calculation
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_6h - ema13_6h
    # Bear Power = Low - EMA13
    bear_power = low_6h - ema13_6h
    
    # Align Elder Ray components to 6h timeframe (already on 6h, but using align for consistency)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # === 6h Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema34 = ema34_1d_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_confirm = volume[i] > vol_ma_aligned[i] * 1.5  # 1.5x average volume
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when: Bear Power becomes positive (selling pressure) OR price closes below EMA34(1d)
            if bear_power_val > 0 or price < ema34:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when: Bull Power becomes negative (buying pressure) OR price closes above EMA34(1d)
            if bull_power_val < 0 or price > ema34:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Bull Power > 0 (buying pressure) AND price > EMA34(1d) (uptrend) AND volume confirmation
            if bull_power_val > 0 and price > ema34 and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Bear Power < 0 (selling pressure) AND price < EMA34(1d) (downtrend) AND volume confirmation
            elif bear_power_val < 0 and price < ema34 and vol_confirm:
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

name = "6h_ElderRay_EMA34Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0