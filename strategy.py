#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_HTFTrend_VolumeSpike_ATRStop_v1
Hypothesis: Donchian(20) breakout on 4h with 1d EMA50 trend filter and volume confirmation (>1.5x average volume). 
Enters long when price breaks above 20-period high with 1d uptrend and volume spike; enters short when price breaks below 20-period low with 1d downtrend and volume spike. 
Exits via ATR-based trailing stop (2.5x ATR) or opposite Donchian breakout. Uses discrete position sizing (0.25) to minimize fee churn. 
Designed to work in both bull and bear markets by following the 1d trend direction and requiring volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian, EMA, ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0  # For trailing stop
    lowest_low_since_entry = 0.0
    base_size = 0.25
    
    # Start after warmup (need 50 for EMA, 20 for Donchian/volume, 14 for ATR)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(upper) or 
            np.isnan(lower) or np.isnan(atr_val)):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Volume confirmation: current volume > 1.5x average volume (strong breakout)
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Long logic: price breaks above Donchian high with 1d uptrend and volume confirmation
        long_condition = (close_val > upper) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Donchian low with 1d downtrend and volume confirmation
        short_condition = (close_val < lower) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic:
        # Long exit: price drops 2.5*ATR from highest high since entry OR price breaks below Donchian low (contrarian signal)
        long_exit = False
        if position == 1:
            highest_high_since_entry = max(highest_high_since_entry, high_val)
            long_exit = (close_val <= highest_high_since_entry - 2.5 * atr_val) or (close_val < lower)
        
        # Short exit: price rises 2.5*ATR from lowest low since entry OR price breaks above Donchian high (contrarian signal)
        short_exit = False
        if position == -1:
            lowest_low_since_entry = min(lowest_low_since_entry, low_val)
            short_exit = (close_val >= lowest_low_since_entry + 2.5 * atr_val) or (close_val > upper)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            highest_high_since_entry = high_val
            lowest_low_since_entry = low_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            highest_high_since_entry = high_val
            lowest_low_since_entry = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_high_since_entry = 0.0
            lowest_low_since_entry = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            highest_high_since_entry = 0.0
            lowest_low_since_entry = 0.0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_HTFTrend_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0