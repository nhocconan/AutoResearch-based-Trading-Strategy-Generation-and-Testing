#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Uses longer timeframe to reduce trade frequency while capturing major trends.
# Requires alignment with 1w EMA50 trend and volume > 2.0x 50-bar average for confirmation.
# ATR(14) trailing stop at 3.0x for wider stops in volatile markets.
# Discrete position sizing at ±0.30 to balance return and risk.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.
# Donchian channels provide clear breakout levels with built-in trend following.
# 1w EMA50 filter ensures we only trade with the major trend, reducing whipsaws.

name = "12h_Donchian20_1wEMA50_VolumeSpike_ATRStop_v1"
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
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) from 1d data for more stable levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian upper and lower bands from 1d data
    # Using 1d data provides more stable breakout levels than lower timeframes
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # ATR(14) for volatility and stoploss
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 2.0x 50-period average (strict to reduce trade frequency)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(50, 50, atr_period, 50) + 1  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, above 1w EMA50, volume confirmation
            if (curr_close > curr_upper and 
                curr_close > curr_ema_50_1w and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Donchian lower, below 1w EMA50, volume confirmation
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_50_1w and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 3.0*ATR from highest point
            if curr_close < highest_since_entry - (3.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 3.0*ATR from lowest point
            if curr_close > lowest_since_entry + (3.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals