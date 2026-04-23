#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper AND 1w EMA50 rising AND volume > 1.5x 20-bar MA.
Short when price breaks below Donchian lower AND 1w EMA50 falling AND volume > 1.5x 20-bar MA.
Exit when price touches opposite Donchian level or 1w EMA50 reverses.
1d timeframe targets 15-40 trades/year to minimize fee drift. Works in bull via breakouts,
in bear via shorting breakdowns with trend filter avoiding counter-trend whipsaws.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_h_exit = np.full(n, np.nan)  # for long exit
    donchian_l_exit = np.full(n, np.nan)  # for short exit
    
    for i in range(20, n):
        # Use lookback of 20 completed bars (i-20 to i-1) to avoid look-ahead
        lookback_high = high[i-20:i]
        lookback_low = low[i-20:i]
        donchian_h[i] = np.max(lookback_high)
        donchian_l[i] = np.min(lookback_low)
        donchian_h_exit[i] = np.max(lookback_high)  # same as upper for exit
        donchian_l_exit[i] = np.min(lookback_low)   # same as lower for exit
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian (needs 20), EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_h[i]
        lower = donchian_l[i]
        upper_exit = donchian_h_exit[i]
        lower_exit = donchian_l_exit[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA50 rising AND volume filter
            if price > upper and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA50 falling AND volume filter
            elif price < lower and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower (opposite) OR EMA50 starts falling
                if price < lower_exit or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper (opposite) OR EMA50 starts rising
                if price > upper_exit or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0