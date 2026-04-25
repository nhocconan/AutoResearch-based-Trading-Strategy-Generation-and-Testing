#!/usr/bin/env python3
"""
1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
Hypothesis: Daily Donchian(20) breakouts capture strong momentum. 1w EMA50 ensures alignment with weekly trend (works in bull via upside breaks, bear via downside breaks). Volume spike (>2.0x 20-day average) confirms conviction. Discrete sizing (0.25) limits fee drift. Target 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume MA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 for Donchian + 1 for shift
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) from previous day
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    donchian_high_20 = high_1d.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = low_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (no shift needed as already 1d)
    donchian_high_aligned = donchian_high_20  # already aligned to 1d bars
    donchian_low_aligned = donchian_low_20
    
    # Calculate 1d 20-period volume MA for volume spike confirmation
    vol_1d = pd.Series(df_1d['volume'])
    vol_ma_20 = vol_1d.rolling(window=20, min_periods=20).mean().shift(1).values
    vol_ma_aligned = vol_ma_20  # already aligned to 1d bars
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian (20+1), volume MA (20+1), and EMA50 warmup
    start_idx = max(21, 21, 50)  # 21 for Donchian/vol MA (20 + 1 for shift), 50 for EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        vol_ma = vol_ma_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-day average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        if position == 0:
            # Long: break above Donchian high + price above 1w EMA50 + volume confirmation
            long_signal = (curr_high > donchian_high) and price_above_ema and volume_confirm
            # Short: break below Donchian low + price below 1w EMA50 + volume confirmation
            short_signal = (curr_low < donchian_low) and price_below_ema and volume_confirm
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below Donchian high OR price crosses below 1w EMA50
            if (curr_close < donchian_high) or (curr_close < ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above Donchian low OR price crosses above 1w EMA50
            if (curr_close > donchian_low) or (curr_close > ema_50_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0