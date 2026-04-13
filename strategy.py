#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h/1d volume confirmation
    # Uses 12h Camarilla levels (R3/S3, R4/S4) for breakout/fade logic
    # Volume confirmation from 1d to avoid false breakouts
    # Discrete sizing (0.25) to minimize fee drag
    # Target: 12-30 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    open_time = prices['open_time']
    
    # Get 12h data for Camarilla pivot calculation (HTF for structure)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla: Pivot = (H + L + C)/3
    # R4 = Pivot + (H-L)*1.1/2, R3 = Pivot + (H-L)*1.1/4
    # S3 = Pivot - (H-L)*1.1/4, S4 = Pivot - (H-L)*1.1/2
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_12h = (prev_high + prev_low + prev_close) / 3.0
    range_12h = prev_high - prev_low
    
    r3_12h = pivot_12h + range_12h * 1.1 / 4.0
    r4_12h = pivot_12h + range_12h * 1.1 / 2.0
    s3_12h = pivot_12h - range_12h * 1.1 / 4.0
    s4_12h = pivot_12h - range_12h * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    # Calculate 1d volume average (20-period) for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_12h_aligned[i]) or 
            np.isnan(r4_12h_aligned[i]) or
            np.isnan(s3_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        idx_1d = i // 4  # 1d bars in 6h timeframe (4 bars per day)
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Breakout conditions: price breaks R4/S4 with volume
        breakout_long = (close[i] > r4_12h_aligned[i]) and volume_confirmed
        breakout_short = (close[i] < s4_12h_aligned[i]) and volume_confirmed
        
        # Fade conditions: price rejects R3/S3 with volume
        fade_long = (close[i] < s3_12h_aligned[i]) and volume_confirmed
        fade_short = (close[i] > r3_12h_aligned[i]) and volume_confirmed
        
        # Stoploss: based on 12h ATR (simplified using daily range from 12h)
        idx_12h = i // 2  # 12h bars in 6h timeframe (2 bars per 12h)
        if idx_12h < len(high_12h) and idx_12h < len(low_12h):
            daily_range_12h = high_12h[idx_12h] - low_12h[idx_12h]
            stop_distance = daily_range_12h * 0.3  # 30% of 12h range
        else:
            stop_distance = 0
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif fade_long and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif fade_short and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "6h_12h_1d_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0