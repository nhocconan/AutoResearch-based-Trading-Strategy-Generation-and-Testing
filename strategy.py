#!/usr/bin/env python3
name = "6h_1d_ElderRay_BullPower_BearPower_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(13) for Elder Ray
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Align daily Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Daily trend filter: EMA(34) on daily close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13, 4)  # Wait for EMA34, EMA13, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema34_1d_aligned[i] > ema34_1d_aligned[i-1]
            
            if bull_power_aligned[i] > 0 and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 with volume and daily downtrend
            elif bear_power_aligned[i] < 0 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power turns negative or volume drops
            if bull_power_aligned[i] <= 0 or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power turns positive or volume drops
            if bear_power_aligned[i] >= 0 or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with daily trend and volume confirmation
# - Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# - Long when Bull Power > 0 (bulls in control) with volume spike and daily uptrend (EMA34 rising)
# - Short when Bear Power < 0 (bears in control) with volume spike and daily downtrend
# - Volume spike (2.0x average) confirms institutional participation in the move
# - Exit when power reverses or volume weakens
# - Works in both bull and bear markets via daily trend filter (EMA34)
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses daily Elder Ray for higher timeframe perspective, avoiding whipsaws
# - Novel combination not recently tested in this session (unlike CCI/RSI/Williams variants)