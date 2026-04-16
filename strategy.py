#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d EMA34 trend filter and volume confirmation
# Long when Bull Power > 0 AND price > 1d EMA34 AND volume > 1.3x 20-period average volume
# Short when Bear Power < 0 AND price < 1d EMA34 AND volume > 1.3x 20-period average volume
# Elder Ray = Bull Power = High - EMA13, Bear Power = Low - EMA13
# EMA13 calculated on 6h data
# Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag on 6h timeframe
# Elder Ray captures bull/bear power relative to trend, effective in both trending and ranging markets
# EMA34 filter ensures alignment with medium-term trend to avoid counter-trend trades
# Volume confirmation adds conviction to signals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d EMA34 (trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 6h EMA13 (for Elder Ray calculation) ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray Components ===
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # === 6h Volume Spike Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_confirm = volume[i] > vol_ma_20[i] * 1.3  # 1.3x average volume for spike
        
        # === EXIT LOGIC: Close position when Elder Ray signal reverses ===
        if position == 1:  # Long position
            # Exit long when Bear Power becomes negative (bearish pressure)
            if bear_val < 0:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit short when Bull Power becomes positive (bullish pressure)
            if bull_val > 0:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Bull Power > 0 (bullish pressure) AND price > 1d EMA34 AND volume spike
            if bull_val > 0 and price > ema_34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Bear Power < 0 (bearish pressure) AND price < 1d EMA34 AND volume spike
            elif bear_val < 0 and price < ema_34_val and vol_confirm:
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

name = "6h_ElderRay_1dEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0