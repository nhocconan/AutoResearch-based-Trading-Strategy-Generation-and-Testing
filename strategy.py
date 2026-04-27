#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter, volume confirmation (>1.5x 20-period average), and ATR-based trailing stop (2.5x ATR). 
Designed for 4h timeframe to achieve 75-200 total trades over 4 years. Works in bull/bear markets by following 12h trend while using Donchian channels for breakout entries.
ATR stoploss controls drawdown during volatile periods. Uses discrete position sizing (0.25) to minimize fee churn.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for volatility and stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 12h EMA50 (50), Donchian (20), ATR (14), volume avg (20)
    start_idx = max(50, 20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_12h_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Donchian channel with 12h trend filter AND volume
            # Long: price breaks above upper channel AND 12h uptrend AND volume
            long_condition = (close_val > upper_channel) and (close_val > ema_val) and vol_conf
            # Short: price breaks below lower channel AND 12h downtrend AND volume
            short_condition = (close_val < lower_channel) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: Donchian lower channel break OR ATR trailing stop (2.5x ATR from highest high since entry)
            # Track highest high since entry for trailing stop
            if i == start_idx or position == 0:
                highest_since_entry = high_val
            else:
                highest_since_entry = max(highest_since_entry, high_val)
            trailing_stop = highest_since_entry - (2.5 * atr_val)
            exit_condition = (close_val < lower_channel) or (close_val < trailing_stop)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Donchian upper channel break OR ATR trailing stop (2.5x ATR from lowest low since entry)
            # Track lowest low since entry for trailing stop
            if i == start_idx or position == 0:
                lowest_since_entry = low_val
            else:
                lowest_since_entry = min(lowest_since_entry, low_val)
            trailing_stop = lowest_since_entry + (2.5 * atr_val)
            exit_condition = (close_val > upper_channel) or (close_val > trailing_stop)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0