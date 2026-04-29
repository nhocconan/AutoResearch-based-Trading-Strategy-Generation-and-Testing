#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d EMA50 trend filter + ATR-based volume confirmation
# Long when close > upper Donchian(20) AND price > 1d EMA50 AND volume > 1.5x ATR(14)-scaled MA
# Short when close < lower Donchian(20) AND price < 1d EMA50 AND volume > 1.5x ATR(14)-scaled MA
# Exit on opposite Donchian level touch (long exit at lower, short exit at upper)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 15-30 trades/year on 6h.
# Donchian channels provide clear breakout levels. EMA50 filters counter-trend moves on 1d.
# Volume confirmation uses ATR scaling to adapt to volatility regimes. Works in both bull and bear markets.

name = "6h_Donchian20_1dEMA50_ATRVolume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for volume normalization
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    upper_donchian = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_donchian = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x ATR-scaled 20-bar volume average
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    atr_scaled_volume_ma = volume_ma_20 * (atr / np.mean(atr[~np.isnan(atr)]))  # Normalize ATR
    volume_confirm = volume > 1.5 * atr_scaled_volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # EMA50, Donchian20, ATR14
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1d_aligned[i]
        upper_don = upper_donchian[i]
        lower_don = lower_donchian[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when close > upper Donchian AND price > 1d EMA50 AND volume confirmation
            if curr_close > upper_don and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when close < lower Donchian AND price < 1d EMA50 AND volume confirmation
            elif curr_close < lower_don and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when close < lower Donchian (opposite level)
            if curr_close < lower_don:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when close > upper Donchian (opposite level)
            if curr_close > upper_don:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals