#!/usr/bin/env python3
"""
4h_MultiSignal_Confluence_v1
Hypothesis: Combine 4h Donchian breakout with 1d ADX trend strength and volume confirmation to capture strong trends while avoiding false breakouts in chop. Designed for low trade frequency (<30/year) to minimize fee drag in both bull and bear markets.
"""

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
    
    # 4h Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ADX for trend strength
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = WilderSmooth(tr, 14)
    dm_plus_14 = WilderSmooth(dm_plus, 14)
    dm_minus_14 = WilderSmooth(dm_minus, 14)
    
    # Avoid division by zero
    di_plus = np.where(tr14 > 0, 100 * dm_plus_14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus_14 / tr14, 0)
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(34, 20)  # ADX needs ~34 periods, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Conditions
        breakout_up = close[i] > donch_high[i]
        breakout_down = close[i] < donch_low[i]
        strong_trend = adx_aligned[i] > 25  # ADX > 25 indicates strong trend
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Long: upward breakout + strong trend + volume
            if breakout_up and strong_trend and vol_ok:
                signals[i] = size
                position = 1
            # Short: downward breakout + strong trend + volume
            elif breakout_down and strong_trend and vol_ok:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: downward breakout OR trend weakens (ADX < 20)
            if breakout_down or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: upward breakout OR trend weakens (ADX < 20)
            if breakout_up or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_MultiSignal_Confluence_v1"
timeframe = "4h"
leverage = 1.0