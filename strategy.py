#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm
Hypothesis: Weekly pivot direction (from 1w data) defines the primary trend. On 6h timeframe, we take Donchian(20) breakouts only in the direction of the weekly pivot trend, with volume confirmation (>1.5x average volume). Uses discrete position sizing (0.25) to minimize fee churn. Designed to work in both bull and bear markets by aligning with the weekly structure, avoiding counter-trend trades that cause whipsaws in ranging conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for Donchian and EMA
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for weekly pivot trend (primary trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # Bias: above pivot = bullish, below pivot = bearish
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_bias = typical_price.values  # Higher = bullish bias
    
    # Align weekly bias to 6h timeframe (completed weekly bars only)
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Load 1d data for additional trend confirmation (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
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
    
    # Start after warmup (need 20 for Donchian, 34 for EMA, 14 for ATR)
    start_idx = max(20, 34, 14)
    
    for i in range(start_idx, n):
        # Hold current position by default
        if position == 0:
            signals[i] = 0.0
        elif position == 1:
            signals[i] = base_size
        else:
            signals[i] = -base_size
        
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(atr[i])):
            continue
        
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_34_1d_aligned[i]
        weekly_bias_val = weekly_bias_aligned[i]
        atr_val = atr[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Weekly trend filter: bullish if price above weekly pivot bias, bearish if below
        # We use the weekly typical price as a dynamic pivot reference
        weekly_bullish = close_val > weekly_bias_val
        weekly_bearish = close_val < weekly_bias_val
        
        # Donchian breakout conditions
        donch_breakout_high = close_val > donch_high
        donch_breakout_low = close_val < donch_low
        
        # Long logic: Donchian breakout above + weekly bullish + 1d uptrend + volume
        long_condition = (donch_breakout_high and weekly_bullish and 
                         close_val > ema_val and volume_confirmed)
        # Short logic: Donchian breakout below + weekly bearish + 1d downtrend + volume
        short_condition = (donch_breakout_low and weekly_bearish and 
                          close_val < ema_val and volume_confirmed)
        
        # Exit logic: Donchian breakout in opposite direction OR trend reversal
        exit_long = donch_breakout_low or close_val < ema_val
        exit_short = donch_breakout_high or close_val > ema_val
        
        # ATR-based stoploss
        if position == 1:
            stop_price = entry_price - atr_multiplier * atr_val
            if close_val < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:
            stop_price = entry_price + atr_multiplier * atr_val
            if close_val > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Execute signals
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # Enter at next bar open, approximate with close
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        # Else hold position (already set above)
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0