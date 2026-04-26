#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dEMA50_Trend_ATRStop_VolumeFilter
Hypothesis: Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation for entry, ATR-based stoploss for exit. 
Designed for low trade frequency (<30/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets. 
Uses discrete position sizing (0.30) to balance profit potential and risk management. Focus on BTC/ETH with volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) from previous closes
    # Use rolling window on close prices for Donchian breakout
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.8x average volume (moderate to reduce trades while keeping signal)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of Donchian (20), EMA50 (50), volume MA (20), ATR (14)
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation and uptrend
            long_signal = (high_val > high_20_val) and (volume_val > 1.8 * vol_ma_val) and (close_val > ema_50_1d_val)
            # Short: price breaks below lower Donchian with volume confirmation and downtrend
            short_signal = (low_val < low_20_val) and (volume_val > 1.8 * vol_ma_val) and (close_val < ema_50_1d_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: ATR stoploss or trend reversal
            if (close_val < entry_price - 2.5 * atr_val or 
                close_val < ema_50_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: ATR stoploss or trend reversal
            if (close_val > entry_price + 2.5 * atr_val or 
                close_val > ema_50_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_ATRStop_VolumeFilter"
timeframe = "4h"
leverage = 1.0