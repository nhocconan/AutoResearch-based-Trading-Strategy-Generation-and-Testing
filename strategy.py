#!/usr/bin/env python3
"""
1d Donchian Breakout with 1w EMA50 Trend Filter and Volume Spike Confirmation
Hypothesis: On daily timeframe, Donchian(20) breakouts aligned with weekly EMA50 trend
and volume confirmation (>2.0x 20-day volume MA) capture strong sustained moves.
In ranging markets (price between Donchian bands), we fade extremes near bands.
Designed for BTC/ETH with 20-40 trades/year to minimize fee drag while maintaining edge
in both bull and bear regimes. Uses 1w HTF for trend filter as specified in experiment.
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
    
    # Get 1w data for EMA50 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 51:  # Need 50 for EMA + 1 for shift
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:  # Need 20 for Donchian + 1 for shift
        return np.zeros(n)
    
    # Previous 20-day high, low for Donchian calculation
    prev_high_20 = df_1d['high'].rolling(20, min_periods=20).max().shift(1).values
    prev_low_20 = df_1d['low'].rolling(20, min_periods=20).min().shift(1).values
    
    # Align to 1d timeframe (prices is already 1d, so alignment is identity but keep for consistency)
    high_20_1d = align_htf_to_ltf(prices, df_1d, prev_high_20)
    low_20_1d = align_htf_to_ltf(prices, df_1d, prev_low_20)
    
    # Calculate 20-period volume MA for volume spike confirmation (1d)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA50, Donchian, and volume MA
    start_idx = max(51, 21, 20)  # 51 for EMA50 (50 + 1 for shift), 21 for Donchian (20 + 1), 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_20_1d[i]) or 
            np.isnan(low_20_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_50_val = ema_50_1w_aligned[i]
        upper_donchian = high_20_1d[i]
        lower_donchian = low_20_1d[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below 1w EMA50
        price_above_ema = curr_close > ema_50_val
        price_below_ema = curr_close < ema_50_val
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        volume_confirm = curr_volume > 2.0 * vol_ma
        
        if position == 0:
            if price_above_ema:
                # Uptrend: look for long breakouts above upper Donchian
                long_signal = (curr_close > upper_donchian) and volume_confirm
            else:
                # Downtrend: look for short breakdowns below lower Donchian
                short_signal = (curr_close < lower_donchian) and volume_confirm
            
            # In ranging markets (price between Donchian bands), fade extremes
            in_range = (curr_close >= lower_donchian) and (curr_close <= upper_donchian)
            if in_range:
                # Fade extremes: long near lower band, short near upper band
                long_signal = (curr_close <= lower_donchian * 1.001) and volume_confirm  # near lower band
                short_signal = (curr_close >= upper_donchian * 0.999) and volume_confirm  # near upper band
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
            # Clear signal flags for next iteration
            if 'long_signal' in locals():
                del long_signal
            if 'short_signal' in locals():
                del short_signal
        elif position == 1:
            # Exit long: price breaks below lower Donchian or reverses below EMA
            if curr_close < lower_donchian or curr_close < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian or reverses above EMA
            if curr_close > upper_donchian or curr_close > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0