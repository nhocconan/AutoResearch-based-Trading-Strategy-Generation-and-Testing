#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeATR_Stop
Hypothesis: Uses 4h Donchian channel breakouts with 1d EMA50 trend filter and volume confirmation for entries.
Enters long when price breaks above 20-period Donchian upper band AND 1d EMA50 up AND volume > 1.5x 20-period average.
Enters short when price breaks below 20-period Donchian lower band AND 1d EMA50 down AND volume > 1.5x 20-period average.
Exits via ATR-based trailing stop (3x ATR from extreme) or opposite Donchian breakout.
Position size: 0.25. Designed for 4h timeframe to balance trade frequency and capture medium-term trends.
Works in both bull and bear markets via trend filter and volatility-adjusted exits.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channel (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # 4h ATR (14-period) for volatility-based stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Tracking variables for trailing stop
    long_entry_price = 0.0
    short_entry_price = 0.0
    long_highest = 0.0
    short_lowest = 0.0
    
    # Warmup: need 1d EMA50 (50), Donchian (20), ATR (14), volume avg (20)
    start_idx = max(50, lookback, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_1d_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1d trend filter AND volume
            # Long: price breaks above upper band AND 1d uptrend AND volume
            long_condition = (close_val > upper_band) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower band AND 1d downtrend AND volume
            short_condition = (close_val < lower_band) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                long_entry_price = close_val
                long_highest = high_val
            elif short_condition:
                signals[i] = -size
                position = -1
                short_entry_price = close_val
                short_lowest = low_val
        elif position == 1:
            # Update highest high for trailing stop
            long_highest = max(long_highest, high_val)
            
            # Exit conditions: ATR trailing stop OR opposite Donchian breakout
            atr_stop = long_highest - (3.0 * atr_val)
            opposite_break = close_val < lower_band
            
            if atr_stop > 0 and close_val <= atr_stop:
                signals[i] = 0.0
                position = 0
            elif opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest low for trailing stop
            short_lowest = min(short_lowest, low_val)
            
            # Exit conditions: ATR trailing stop OR opposite Donchian breakout
            atr_stop = short_lowest + (3.0 * atr_val)
            opposite_break = close_val > upper_band
            
            if atr_stop > 0 and close_val >= atr_stop:
                signals[i] = 0.0
                position = 0
            elif opposite_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeATR_Stop"
timeframe = "4h"
leverage = 1.0