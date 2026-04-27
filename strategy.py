#!/usr/bin/env python3
"""
6h_ADX_DMI_VolumeSpike_1dTrend
Hypothesis: ADX > 25 with +DI > -DI for long, -DI > +DI for short on 6h, combined with 1d EMA50 trend filter and volume confirmation. 
ADX measures trend strength, DMI shows direction. Volume spike confirms conviction. Works in both bull/bear by only taking strong directional moves with confirmation.
Target: 12-30 trades/year on 6h to minimize fee drag.
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
    
    # Calculate ADX and DMI (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR
    
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # 25% position size
    
    # Warmup: need enough for ADX/DMI, EMA50, volume average
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Flat - look for entry: ADX > 25 (strong trend), DMI shows direction, aligned with 1d trend, volume spike
            long_condition = adx[i] > 25 and plus_di[i] > minus_di[i] and close_val > ema_trend and vol_spike
            short_condition = adx[i] > 25 and minus_di[i] > plus_di[i] and close_val < ema_trend and vol_spike
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when trend weakens (ADX < 20) or DMI crosses or 1d trend turns down
            if adx[i] < 20 or minus_di[i] > plus_di[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when trend weakens (ADX < 20) or DMI crosses or 1d trend turns up
            if adx[i] < 20 or plus_di[i] > minus_di[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_DMI_VolumeSpike_1dTrend"
timeframe = "6h"
leverage = 1.0