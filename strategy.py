#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Daily Donchian(20) breakout with 1-week EMA34 trend filter and volume confirmation (>2x average volume).
Uses discrete position sizing (0.25) to minimize fee churn. Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
Designed to work in both bull and bear markets via 1w trend alignment and strict volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for indicators
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period SMA)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    atr_multiplier = 2.5  # ATR stoploss multiplier
    
    # Start after warmup (need 20 for Donchian, 34 for EMA, 14 for ATR, 20 for volume)
    start_idx = max(20, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Hold current position by default
        if position == 0:
            signals[i] = 0.0
        elif position == 1:
            signals[i] = base_size
        else:
            signals[i] = -base_size
        
        # Skip if any data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_34_1w_aligned[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(atr[i]):
            continue
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1w_aligned[i]
        atr_val = atr[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Long logic: price breaks above upper Donchian with 1w uptrend and volume confirmation
        long_condition = (close_val > upper_channel) and (close_val > ema_val) and volume_confirmed
        # Short logic: price breaks below lower Donchian with 1w downtrend and volume confirmation
        short_condition = (close_val < lower_channel) and (close_val < ema_val) and volume_confirmed
        
        # Exit logic: trend reversal (price crosses 1w EMA34)
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0