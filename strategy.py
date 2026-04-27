#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: Uses 4h Donchian(20) breakout for entries with 1d EMA34 trend filter and volume confirmation.
Exits via ATR-based trailing stop (3*ATR from extreme) or trend reversal.
Designed for 4h timeframe to achieve 75-200 total trades over 4 years with low fee drag.
Works in both bull and bear markets by following 1d trend while using Donchian breakouts for precise entries.
ATR stoploss controls drawdown during volatile periods.
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
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for volatility and stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Track extreme prices for trailing stop
    long_entry_price = 0.0
    short_entry_price = 0.0
    long_highest = 0.0
    short_lowest = 0.0
    
    # Warmup: need Donchian(20), EMA34(34), ATR(14), volume avg(20)
    start_idx = max(lookback, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Donchian channel with 1d trend filter AND volume
            # Long: price breaks above upper channel AND 1d uptrend AND volume
            long_condition = (close_val > upper_channel) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower channel AND 1d downtrend AND volume
            short_condition = (close_val < lower_channel) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                long_entry_price = close_val
                long_highest = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                short_entry_price = close_val
                short_lowest = close_val
        elif position == 1:
            # Update highest price for trailing stop
            long_highest = max(long_highest, close_val)
            
            # Exit conditions:
            # 1. ATR trailing stop: price drops 3*ATR from highest high
            # 2. Trend reversal: price closes below 1d EMA34
            stop_price = long_highest - 3.0 * atr_val
            trend_exit = close_val < ema_val
            
            if (close_val <= stop_price) or trend_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Update lowest price for trailing stop
            short_lowest = min(short_lowest, close_val)
            
            # Exit conditions:
            # 1. ATR trailing stop: price rises 3*ATR from lowest low
            # 2. Trend reversal: price closes above 1d EMA34
            stop_price = short_lowest + 3.0 * atr_val
            trend_exit = close_val > ema_val
            
            if (close_val >= stop_price) or trend_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0