#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily data for Elder Ray calculation ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Exponential Moving Average (13-period) ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray: Bull Power and Bear Power ===
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # === Smooth Elder Ray with EMA (6-period) to reduce noise ===
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # === Align smoothed Elder Ray to 6h timeframe ===
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === ATR for volatility filter (14-period on 6h) ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_6h = pd.Series(atr_6h).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 200
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_ma_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        bull_power_val = bull_power_6h[i]
        bear_power_val = bear_power_6h[i]
        vol_spike = volume_spike[i]
        atr_ma = atr_ma_6h[i]
        
        # === EXIT LOGIC: Exit when power weakens or volatility drops ===
        if position == 1:  # Long position
            # Exit when bull power turns negative or volatility drops
            if bull_power_val < 0 or atr_ma < (atr_6h[i] * 0.5):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when bear power turns positive or volatility drops
            if bear_power_val > 0 or atr_ma < (atr_6h[i] * 0.5):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Strong bull power with volume spike and sufficient volatility
            if bull_power_val > 0 and vol_spike and atr_ma > 0:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Strong bear power with volume spike and sufficient volatility
            elif bear_power_val < 0 and vol_spike and atr_ma > 0:
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

name = "6h_ElderRay_VolumeSpike_ATRFilter"
timeframe = "6h"
leverage = 1.0