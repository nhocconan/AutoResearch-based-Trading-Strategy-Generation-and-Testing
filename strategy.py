#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above 20-period high, 12h EMA50 up-trend, volume > 2.0x average
# Short when price breaks below 20-period low, 12h EMA50 down-trend, volume > 2.0x average
# Exit via ATR-based trailing stop (3*ATR from extreme) or midpoint reversal
# Uses discrete position sizing (0.25) and strong volume filter to target 20-50 trades/year.
# Designed to work in both bull and bear markets by following the higher timeframe trend.

name = "4h_Donchian20_12hEMA50_Volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR(14) on 4h for stoploss
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 4h indicators to 4h timeframe (no additional delay needed)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 50)  # Volume and 12h EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_dch_high = donchian_high_aligned[i]
        curr_dch_low = donchian_low_aligned[i]
        curr_dch_mid = donchian_mid_aligned[i]
        curr_ema50_12h = ema_50_12h_aligned[i]
        curr_atr = atr_14_aligned[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle position management and exits
        if position == 1:  # Long position
            # Update highest since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions:
            # 1. ATR trailing stop: price < highest - 3*ATR
            # 2. Mean reversion: price < midpoint
            if curr_close < (highest_since_entry - 3.0 * curr_atr) or curr_close < curr_dch_mid:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions:
            # 1. ATR trailing stop: price > lowest + 3*ATR
            # 2. Mean reversion: price > midpoint
            if curr_close > (lowest_since_entry + 3.0 * curr_atr) or curr_close > curr_dch_mid:
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average (strong filter)
            vol_confirmed = curr_volume > 2.0 * curr_vol_ma
            
            # Long when price breaks above Donchian high, 12h EMA50 up-trend, volume confirmed
            if curr_high > curr_dch_high and curr_close > curr_ema50_12h and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short when price breaks below Donchian low, 12h EMA50 down-trend, volume confirmed
            elif curr_low < curr_dch_low and curr_close < curr_ema50_12h and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
    
    return signals