#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day volume confirmation and ATR volatility filter
# Works in bull markets by capturing breakouts; works in bear markets by avoiding low-volatility false signals
# Volume spike filters out low-conviction moves; ATR filter ensures trades occur in sufficient volatility regimes
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 4h Donchian channel (20-period) ===
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # === 1d ATR(14) for volatility filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # === 1d volume confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 4h volume confirmation ===
    vol_ma_15_4h = pd.Series(volume).rolling(window=15, min_periods=15).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_ma_15_4h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 1d volume (avoid calling get_htf_data in loop)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume spike: current 1d volume > 2.0x 20-period average AND 4h volume > 1.5x 15-period average
        vol_spike_1d = volume_1d_aligned[i] > vol_ma_20_1d_aligned[i] * 2.0
        vol_spike_4h = volume[i] > vol_ma_15_4h[i] * 1.5
        
        # Donchian breakout conditions
        breakout_up = close[i] > high_20[i-1]  # Break above previous period's high
        breakout_down = close[i] < low_20[i-1]  # Break below previous period's low
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-100):i])
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Donchian breakout up + volume spike + volatility filter
            if breakout_up and vol_spike_1d and vol_spike_4h and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Donchian breakout down + volume spike + volatility filter
            elif breakout_down and vol_spike_1d and vol_spike_4h and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long when price returns to middle of channel or volatility drops
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] < mid_channel or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price returns to middle of channel or volatility drops
            mid_channel = (high_20[i] + low_20[i]) / 2
            if close[i] > mid_channel or not vol_filter:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Vol2.0x_1.5x"
timeframe = "4h"
leverage = 1.0