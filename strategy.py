#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1w trend filter and volume confirmation.
    # Donchian breakouts capture momentum; weekly trend ensures direction alignment.
    # Volume confirmation filters false breakouts. Works in bull/bear via weekly trend.
    # Target: 50-150 total trades over 4 years (12-37/year). Discrete size 0.25 to minimize fees.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian and volume (call ONCE before loop)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1w data for trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 6h Donchian Channel (20-period)
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Upper band: highest high over 20 periods
    upper_channel = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over 20 periods
    lower_channel = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume mean (20-period) with min_periods
    volume_6h_series = pd.Series(df_6h['volume'].values)
    vol_ma_20_6h = volume_6h_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_6h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_6h, lower_channel)
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 6h volume for spike detection
        volume_6h_raw = df_6h['volume'].values
        vol_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h_raw)
        
        # Volume filter: current 6h volume > 1.5 * 20-period mean (volume spike)
        volume_confirmation = vol_6h_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_aligned[i]
        
        # Donchian breakout conditions
        breakout_upper = high[i] > upper_channel_aligned[i]
        breakout_lower = low[i] < lower_channel_aligned[i]
        
        # Entry conditions
        long_entry = breakout_upper and volume_confirmation and price_above_weekly_ema
        short_entry = breakout_lower and volume_confirmation and price_below_weekly_ema
        
        # Exit conditions: opposite Donchian breakout or loss of volume confirmation
        long_exit = breakout_lower or not volume_confirmation
        short_exit = breakout_upper or not volume_confirmation
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_6h_1w_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0