#!/usr/bin/env python3
"""
6h ADX + Williams Alligator Breakout with Volume Confirmation
Hypothesis: ADX > 25 identifies strong trends, Williams Alligator (SMAs with specific periods) provides entry/exit signals, and volume confirmation filters false breakouts. Works in both bull/bear markets by capturing strong directional moves while avoiding choppy regimes. Discrete position sizing (0.25) targets ~50-100 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13-period SMA, shifted 8), Teeth (8-period SMA, shifted 5), Lips (5-period SMA, shifted 3)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # ADX calculation (14-period)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    di_plus = 100 * (dm_plus_14 / tr14)
    di_minus = 100 * (dm_minus_14 / tr14)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (vol_ma * 1.8)
    
    # 12h EMA50 trend filter (MTF) - loaded ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(13, 8, 5, 14, 30, 50) + 8  # max periods + jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(di_plus[i]) or np.isnan(di_minus[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_conf = volume_confirm[i]
        
        # Alligator conditions: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # ADX condition: strong trend
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + ADX > 25 + volume confirmation + 12h EMA50 trend alignment
            long_entry = bullish_alligator and strong_trend and vol_conf and (curr_close > ema_50_12h_aligned[i])
            short_entry = bearish_alligator and strong_trend and vol_conf and (curr_close < ema_50_12h_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Alligator death cross (Lips < Teeth) OR ADX weakens (< 20)
            if lips[i] < teeth[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator death cross (Lips > Teeth) OR ADX weakens (< 20)
            if lips[i] > teeth[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_WilliamsAlligator_Breakout_VolumeConfirm_12hEMA50"
timeframe = "6h"
leverage = 1.0