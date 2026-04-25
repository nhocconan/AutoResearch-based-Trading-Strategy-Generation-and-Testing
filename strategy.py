#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d EMA34 Trend + Volume Spike + ATR Stoploss
Hypothesis: Donchian channel breakouts capture strong momentum moves. 
Filtered by 1d EMA34 for primary trend alignment and volume spike for confirmation.
Works in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
Uses ATR-based trailing stop and discrete position sizing (0.0, ±0.25) to minimize fee churn.
Target: 25-40 trades/year on 4h timeframe (~100-160 total over 4 years).
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
    
    # Get 1d data for EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for stoploss and volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    # Using rolling window with min_periods to avoid look-ahead
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.5 * 20-period average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop on longs
    lowest_since_entry = 0.0   # for trailing stop on shorts
    
    # Start index: need enough for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        atr_val = atr[i]
        donchian_upper = highest_high[i]
        donchian_lower = lowest_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above Donchian upper AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_close > donchian_upper) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below Donchian lower AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_close < donchian_lower) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            highest_since_entry = max(highest_since_entry, curr_high)
            
            # Exit conditions:
            # 1. Price crosses below Donchian lower (breakdown)
            # 2. Trailing stop: price drops 2.0*ATR from highest since entry
            # 3. Price crosses below EMA (trend change)
            exit_signal = (curr_close < donchian_lower) or \
                         (curr_close < highest_since_entry - 2.0 * atr_val) or \
                         (curr_close < ema_trend)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Exit conditions:
            # 1. Price crosses above Donchian upper (breakout)
            # 2. Trailing stop: price rises 2.0*ATR from lowest since entry
            # 3. Price crosses above EMA (trend change)
            exit_signal = (curr_close > donchian_upper) or \
                         (curr_close > lowest_since_entry + 2.0 * atr_val) or \
                         (curr_close > ema_trend)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0