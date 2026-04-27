#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: Uses 4h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above upper band AND 1d close > EMA50 (uptrend) AND volume > 2.0 * 20-period average.
Short when price breaks below lower band AND 1d close < EMA50 (downtrend) AND volume > 2.0 * 20-period average.
Exit via ATR-based trailing stop: signal→0 when price < highest_high - 3*ATR (long) or price > lowest_low + 3*ATR (short).
Designed for 4h timeframe to achieve 75-200 total trades over 4 years with low fee drag.
Works in both bull and bear markets by following 1d trend while using Donchian breakouts for precise entries.
ATR stoploss controls drawdown during volatile periods.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channel (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Track highest high since entry for trailing stop (long)
    # Track lowest low since entry for trailing stop (short)
    highest_since_entry = np.full(n, np.nan)
    lowest_since_entry = np.full(n, np.nan)
    
    # Warmup: need EMA50 (50), Donchian (20), ATR (14), volume avg (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Donchian channel with 1d trend filter AND volume
            # Long: price breaks above upper band AND 1d uptrend AND volume
            long_condition = (close_val > upper_band) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower band AND 1d downtrend AND volume
            short_condition = (close_val < lower_band) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                highest_since_entry[i] = high_val  # Initialize trailing stop
            elif short_condition:
                signals[i] = -size
                position = -1
                lowest_since_entry[i] = low_val  # Initialize trailing stop
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high_val)
            # Exit long when price drops below highest_high - 3*ATR (trailing stop)
            exit_condition = close_val < (highest_since_entry[i] - 3.0 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                highest_since_entry[i] = np.nan  # Reset
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low_val)
            # Exit short when price rises above lowest_low + 3*ATR (trailing stop)
            exit_condition = close_val > (lowest_since_entry[i] + 3.0 * atr_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
                lowest_since_entry[i] = np.nan  # Reset
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0