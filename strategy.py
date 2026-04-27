#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeATR_Stop
Hypothesis: Uses 4h Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar average).
Enter long when price breaks above 20-period high AND 1d close > EMA50 AND volume > 1.5x average.
Enter short when price breaks below 20-period low AND 1d close < EMA50 AND volume > 1.5x average.
Exit via ATR-based trailing stop: long exits when price < highest high since entry - 2.5*ATR(14);
short exits when price > lowest low since entry + 2.5*ATR(14).
ATR stoploss respects engine semantics by using only close-based exits (no intrabar simulation).
Designed for 4h timeframe to work in both bull and bear markets via trend filter and volatility-adjusted stops.
Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position size.
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
    
    # ATR(14) for volatility and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first bar
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Track extreme prices for trailing stop
    long_highest_high = np.full(n, np.nan)
    short_lowest_low = np.full(n, np.nan)
    
    # Warmup: need 1d EMA50 (50), Donchian (20), ATR (14), volume avg (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Donchian breakout with 1d trend filter AND volume
            # Long: price breaks above upper channel AND 1d uptrend AND volume
            long_condition = (close_val > upper_channel) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower channel AND 1d downtrend AND volume
            short_condition = (close_val < lower_channel) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                long_highest_high[i] = high_val  # initialize trailing stop
            elif short_condition:
                signals[i] = -size
                position = -1
                short_lowest_low[i] = low_val  # initialize trailing stop
        elif position == 1:
            # Update highest high for trailing stop
            long_highest_high[i] = max(long_highest_high[i-1], high_val)
            # Exit long when price trails from highest high by 2.5*ATR
            exit_level = long_highest_high[i] - (2.5 * atr_val)
            exit_condition = close_val < exit_level
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest low for trailing stop
            short_lowest_low[i] = min(short_lowest_low[i-1], low_val)
            # Exit short when price trails above lowest low by 2.5*ATR
            exit_level = short_lowest_low[i] + (2.5 * atr_val)
            exit_condition = close_val > exit_level
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeATR_Stop"
timeframe = "4h"
leverage = 1.0