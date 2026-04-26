#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts with 1d trend filter and volume confirmation capture strong momentum moves across market regimes. 
In bull markets: price breaks above 20-period high with 1d uptrend and volume spike → long. 
In bear markets: price breaks below 20-period low with 1d downtrend and volume spike → short. 
Uses 1d EMA34 for trend (more responsive than weekly) and ATR-based stoploss to limit drawdown. 
Target: 75-200 trades over 4 years (19-50/year). Discrete position sizing (0.0, ±0.25) minimizes fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need 34 for EMA and 20 for Donchian
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) - calculated from price history
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need 34 for EMA and 20 for Donchian)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Update highest/lowest since entry for trailing stop
        if position == 1:
            highest_since_entry = max(highest_since_entry, high[i])
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Calculate Donchian levels for current bar (using lookback of 20 completed bars)
        lookback_start = max(0, i - 19)
        lookback_end = i + 1
        highest_20 = np.max(high[lookback_start:lookback_end])
        lowest_20 = np.min(low[lookback_start:lookback_end])
        
        # Long logic: price breaks above 20-period high with volume spike and 1d uptrend
        long_breakout = close_val > highest_20
        long_condition = long_breakout and vol_spike and (close_val > ema_val)
        
        # Short logic: price breaks below 20-period low with volume spike and 1d downtrend
        short_breakout = close_val < lowest_20
        short_condition = short_breakout and vol_spike and (close_val < ema_val)
        
        # Stoploss logic: ATR-based trailing stop
        stop_long = False
        stop_short = False
        if position == 1 and highest_since_entry > 0:
            stop_long = close_val < (highest_since_entry - 2.5 * atr[i])
        elif position == -1 and lowest_since_entry > 0:
            stop_short = close_val > (lowest_since_entry + 2.5 * atr[i])
        
        # Exit logic: trend reversal (secondary exit)
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
            entry_price = close_val
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
            entry_price = close_val
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
        elif position == 1 and (stop_long or exit_long):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        elif position == -1 and (stop_short or exit_short):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0