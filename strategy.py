#!/usr/bin/env python3
"""
6h Elder Ray + ADX + Volume Spike Strategy
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) 
measures bull/bear strength relative to trend. Combined with ADX for trend strength 
and volume spikes for confirmation, this captures strong directional moves while 
avoiding chop. Works in bull (strong bull power) and bear (strong bear power) 
markets. Low trade frequency via strict ADX>25 and volume>2x average filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for EMA and ADX calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate EMA13 for Elder Ray
    ema13 = pd.Series(close_daily).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high_daily - ema13
    bear_power = low_daily - ema13
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = np.abs(high_daily - low_daily)
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.where((high_daily - np.roll(high_daily, 1)) > (np.roll(low_daily, 1) - low_daily),
                       np.maximum(high_daily - np.roll(high_daily, 1), 0), 0)
    minus_dm = np.where((np.roll(low_daily, 1) - low_daily) > (high_daily - np.roll(high_daily, 1)),
                        np.maximum(np.roll(low_daily, 1) - low_daily, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align daily indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_daily, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_daily, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        adx_val = adx_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0x 20-period average
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_ok = vol_current > 2.0 * vol_ma
        
        if position == 0:
            # Long entry: strong bull power + strong trend + volume
            if bull_power_val > 0 and adx_val > 25 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short entry: strong bear power + strong trend + volume
            elif bear_power_val < 0 and adx_val > 25 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative or trend weakens
            if bull_power_val <= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive or trend weakens
            if bear_power_val >= 0 or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_ADX_VolumeSpike"
timeframe = "6h"
leverage = 1.0