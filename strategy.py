#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour strategy using 1-day Donchian breakout with volume confirmation and ATR stop.
# In trending markets, trade breakouts in the direction of the 1-day trend (price > EMA50).
# Uses volume > 1.5x 20-period average to confirm momentum.
# Position size: 0.25 (25%) to balance risk and return.
# Target: 20-50 trades per year per symbol to minimize fee drag.
# Works in both bull and bear markets by filtering trades with trend and volume.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day EMA(50) for trend direction
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day Donchian channels (20-period high/low)
    donch_len = 20
    if len(df_1d) < donch_len:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Rolling max/min for Donchian channels
    dh_1d = pd.Series(high_1d).rolling(window=donch_len, min_periods=donch_len).max().values
    dl_1d = pd.Series(low_1d).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Align Donchian levels to 12h timeframe
    dh_1d_aligned = align_htf_to_ltf(prices, df_1d, dh_1d)
    dl_1d_aligned = align_htf_to_ltf(prices, df_1d, dl_1d)
    
    # ATR for volatility and stop loss (1-day ATR)
    atr_len = 14
    if len(df_1d) < atr_len:
        return np.zeros(n)
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1]) if len(close_1d) > 1 else np.array([])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1]) if len(close_1d) > 1 else np.array([])
    
    if len(tr1) > 0 and len(tr2) > 0 and len(tr3) > 0:
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
    else:
        # Handle edge case with insufficient data
        tr = np.full(len(high_1d), np.nan)
        if len(high_1d) > 0:
            tr[0] = np.nan
    
    atr_1d = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(ema_len*2, donch_len, atr_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(dh_1d_aligned[i]) or
            np.isnan(dl_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1-day EMA50
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts in direction of 1-day trend
            if bullish_trend and volume_confirmed:
                # Long breakout above Donchian high
                if close[i] > dh_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
            elif bearish_trend and volume_confirmed:
                # Short breakdown below Donchian low
                if close[i] < dl_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian low or stops hit
            if close[i] < dl_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian high or stops hit
            if close[i] > dh_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_EMA_Volume_v1"
timeframe = "12h"
leverage = 1.0