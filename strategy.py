#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using Camarilla pivot breakout with volume confirmation and 1w trend filter
    # Uses 1d Camarilla levels (R3/S3 for mean reversion fade, R4/S4 for breakout continuation)
    # 1w EMA filter ensures alignment with higher timeframe trend
    # Volume confirmation reduces false breakouts
    # Discrete sizing (0.25) to minimize fee drag
    # Target: 12-25 trades/year (50-100 over 4 years) to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot points: based on previous day's range
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA (50-period)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation (20-period average)
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 6h primary timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(r4_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        # Find the 1d index for current 6h bar (approximate: 4x 6h bars per day)
        idx_1d = i // 4
        if idx_1d >= len(volume_1d):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_1d[idx_1d] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: price relative to 1w EMA
        price_above_ema = close[i] > ema_50_1w_aligned[i]
        price_below_ema = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions:
        # Mean reversion fade at R3/S3: price reverses from extreme levels
        # Breakout continuation at R4/S4: price breaks and continues with volume
        fade_long = (close[i] <= s3_1d_aligned[i]) and price_above_ema and volume_confirmed
        fade_short = (close[i] >= r3_1d_aligned[i]) and price_below_ema and volume_confirmed
        breakout_long = (close[i] >= r4_1d_aligned[i]) and price_above_ema and volume_confirmed
        breakout_short = (close[i] <= s4_1d_aligned[i]) and price_below_ema and volume_confirmed
        
        enter_long = fade_long or breakout_long
        enter_short = fade_short or breakout_short
        
        # Stoploss: based on 1d ATR approximation
        if idx_1d < len(high_1d) and idx_1d < len(low_1d):
            daily_range = high_1d[idx_1d] - low_1d[idx_1d]
            stop_distance = daily_range * 0.3  # 30% of daily range
        else:
            stop_distance = 0
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
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

name = "6h_1d_1w_camarilla_pivot_breakout_fade_volume_v1"
timeframe = "6h"
leverage = 1.0