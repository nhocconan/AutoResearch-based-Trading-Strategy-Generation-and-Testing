#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above 20-period high and 1d EMA34 is rising (bullish trend).
# Enter short when price breaks below 20-period low and 1d EMA34 is falling (bearish trend).
# Volume must be > 1.8x 20-bar average for confirmation to avoid false breakouts.
# ATR(14) trailing stop at 2.5x for risk management. Discrete position sizing at ±0.25.
# Target: 50-150 total trades over 4 years (12-37/year). Works in both bull and bear markets
# by requiring 1d trend alignment to avoid counter-trend whipsaws and using Donchian channels
# for structured breakouts with volume confirmation.

name = "12h_Donchian20_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h data
    highest_high_20 = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to primary timeframe (12h -> 12h: identity but using helper for consistency)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_12h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_20)
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA34 slope for trend direction (rising/falling)
    ema_34_slope = np.zeros_like(ema_34_1d_aligned)
    ema_34_slope[1:] = ema_34_1d_aligned[1:] - ema_34_1d_aligned[:-1]
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 34, atr_period, 20) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(highest_high_20_aligned[i]) or
            np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_34_slope[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_high = highest_high_20_aligned[i]
        curr_lowest_low = lowest_low_20_aligned[i]
        curr_ema_34 = ema_34_1d_aligned[i]
        curr_ema_34_slope = ema_34_slope[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-period high, EMA34 rising, volume confirmation
            if (curr_close > curr_highest_high and 
                curr_ema_34_slope > 0 and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below 20-period low, EMA34 falling, volume confirmation
            elif (curr_close < curr_lowest_low and 
                  curr_ema_34_slope < 0 and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest point
            if curr_close < highest_since_entry - (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest point
            if curr_close > lowest_since_entry + (2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals