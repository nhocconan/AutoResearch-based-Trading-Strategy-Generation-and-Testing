#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above upper Donchian AND price > 1d EMA34 AND volume > 1.8x 20-bar avg
# Short when price breaks below lower Donchian AND price < 1d EMA34 AND volume > 1.8x 20-bar avg
# Exit when price crosses opposite Donchian band (mean reversion to median)
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 20-50 trades total over 4 years (5-12/year) on 4h to minimize fee drag.
# Donchian provides structure; 1d EMA34 filters counter-trend moves in bear markets.
# Volume spike ensures institutional participation, reducing false breakouts.
# Works in both bull (trend continuation) and bear (mean reversion within trend) regimes.

name = "4h_Donchian20_1dEMA34_VolumeConfirm_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 4h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels from previous 4h bar's OHLC (20-bar lookback)
    # Need to align 4h OHLC to 4h bars (trivial since primary is 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Extract 4h OHLC values
    fourh_high = df_4h['high'].values
    fourh_low = df_4h['low'].values
    fourh_close = df_4h['close'].values
    
    # Align 4h OHLC to 4h timeframe (no shift needed as both are 4h)
    fourh_high_aligned = align_htf_to_ltf(prices, df_4h, fourh_high)
    fourh_low_aligned = align_htf_to_ltf(prices, df_4h, fourh_low)
    fourh_close_aligned = align_htf_to_ltf(prices, df_4h, fourh_close)
    
    # Calculate Donchian channels (20-bar high/low) for each 4h bar based on prior 20 bars
    # We need to look back 20 bars from the current bar, so we shift the aligned arrays by 1
    # to use only completed prior bars
    lookback = 20
    # Shift by 1 to use only completed prior bars (lookback period ends at prior bar)
    shifted_high = np.roll(fourh_high_aligned, 1)
    shifted_low = np.roll(fourh_low_aligned, 1)
    # Set first value to NaN as we don't have 20 prior bars
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
    
    start_idx = max(34, 20) + 1  # EMA34 and Donchian lookback warmup + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1d_aligned[i]
        
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
            # Long when price breaks above upper Donchian AND price > 1d EMA34 AND volume confirmation
            if curr_close > upper and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian AND price < 1d EMA34 AND volume confirmation
            elif curr_close < lower and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals