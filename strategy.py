#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop_v1
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation (>1.5x average volume). Uses ATR-based stoploss (2.0*ATR) and discrete position sizing (0.25) to minimize fee churn. Works in both bull and bear markets by following the 1-week trend direction, confirmed by volume to avoid false breakouts. ATR stop reduces whipsaw vs pure retest exit. Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need warmup for Donchian, EMA, volume, ATR
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels from historical daily data
    # We need 20 days of lookback, so we'll use rolling window on daily data
    # For simplicity, we'll approximate using 20*24 = 480 hours, but since we're on 1d timeframe:
    # We'll use 20-period lookback on the daily closes
    high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.0
    
    # Start after warmup (need 50 for EMA, 20 for Donchian, 20 for volume, 14 for ATR)
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Get current values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1w_aligned[i]
        upper_channel = high_20[i]
        lower_channel = low_20[i]
        atr_val = atr[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(upper_channel) or 
            np.isnan(lower_channel) or np.isnan(atr_val)):
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
        
        # Long logic: price breaks above Donchian upper channel with 1w uptrend and volume confirmation
        long_condition = (close_val > upper_channel) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below Donchian lower channel with 1w downtrend and volume confirmation
        short_condition = (close_val < lower_channel) and (close_val < ema_val) and volume_confirmed
        
        # Stoploss logic: price moves against position by atr_multiplier * ATR from entry
        long_stop = (position == 1 and close_val < entry_price - atr_multiplier * atr_val)
        short_stop = (position == -1 and close_val > entry_price + atr_multiplier * atr_val)
        
        # Exit logic: 
        # Long exit: price retests or breaks below Donchian lower channel (failed breakout) OR stoploss hit
        long_exit = (position == 1 and (close_val <= lower_channel or long_stop))
        # Short exit: price retests or breaks above Donchian upper channel (failed breakout) OR stoploss hit
        short_exit = (position == -1 and (close_val >= upper_channel or short_stop))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0