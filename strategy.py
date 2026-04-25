#!/usr/bin/env python3
"""
6h Elder Ray Power + ADX Regime + Volume Spike
Hypothesis: Elder Ray Bull/Bear Power measures buying/selling pressure relative to EMA13.
In ranging markets (ADX < 20), fade extreme power readings. In trending markets (ADX > 25),
breakouts in direction of power with volume confirmation capture institutional moves.
Uses 1d EMA34 for higher timeframe trend filter. Target: 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA13 for Elder Ray calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # ADX calculation (14-period)
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), 
                    np.abs(low - np.roll(close, 1)))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    atr14_safe = np.where(atr14 == 0, 1e-10, atr14)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr14_safe
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr14_safe
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    # 1d EMA34 trend filter (MTF)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(20, 34, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Regime filter: ADX < 20 = ranging, ADX > 25 = trending
            if adx[i] < 20:
                # Ranging market: fade extreme power readings
                long_entry = (bull_power[i] < -0.5 * np.std(bull_power[max(0,i-50):i+1])) and vol_spike and (curr_close > ema_34_1d_aligned[i])
                short_entry = (bear_power[i] > 0.5 * np.std(bear_power[max(0,i-50):i+1])) and vol_spike and (curr_close < ema_34_1d_aligned[i])
            elif adx[i] > 25:
                # Trending market: breakouts in direction of power
                long_entry = (bull_power[i] > 0.5 * np.std(bull_power[max(0,i-50):i+1])) and vol_spike and (curr_close > ema_34_1d_aligned[i])
                short_entry = (bear_power[i] < -0.5 * np.std(bear_power[max(0,i-50):i+1])) and vol_spike and (curr_close < ema_34_1d_aligned[i])
            else:
                # Transition zone: no entries
                long_entry = False
                short_entry = False
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: bear power turns negative or ADX weakens
            if bear_power[i] < 0 or adx[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull power turns positive or ADX weakens
            if bull_power[i] > 0 or adx[i] < 18:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_ADX_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0