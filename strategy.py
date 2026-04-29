#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above upper Donchian AND price > 1w EMA50 AND volume > 1.8x 20-bar avg
# Short when price breaks below lower Donchian AND price < 1w EMA50 AND volume > 1.8x 20-bar avg
# Exit when price crosses opposite Donchian band (mean reversion to median)
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 20-50 trades total over 4 years (5-12/year) on 1d to minimize fee drag.
# Donchian provides structure; 1w EMA50 filters counter-trend moves in bear markets.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in both bull (trend continuation) and bear (mean reversion within trend) regimes.

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous day's OHLC (20-day lookback)
    # Need to align daily OHLC to 1d bars (trivial since primary is 1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Extract daily OHLC values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Align daily OHLC to 1d timeframe (no shift needed as both are 1d)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, daily_close)
    
    # Calculate Donchian channels (20-day high/low) for each 1d bar based on prior 20 days
    # We need to look back 20 days from the current bar, so we shift the aligned arrays by 1
    # to use only completed prior days
    lookback = 20
    # Shift by 1 to use only completed prior days (lookback period ends at prior day)
    shifted_high = np.roll(daily_high_aligned, 1)
    shifted_low = np.roll(daily_low_aligned, 1)
    # Set first value to NaN as we don't have 20 prior days
    shifted_high[0] = np.nan
    shifted_low[0] = np.nan
    
    # Calculate rolling max/min over the lookback period
    upper_channel = pd.Series(shifted_high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(shifted_low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 1  # EMA50 and Donchian lookback warmup + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1w_aligned[i]
        
        # Donchian levels
        upper = upper_channel[i]
        lower = lower_channel[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below lower Donchian (mean reversion)
            if curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above upper Donchian (mean reversion)
            if curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper Donchian AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals