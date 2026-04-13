#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h volume spike + 1d ADX trend filter
    # Designed for low trade frequency (20-40/year) to minimize fee drag on 4h timeframe
    # Uses 12h/1d for signal validation, 4h only for entry/exit timing
    # Works in both bull and bear: breakout captures momentum, ADX filter avoids chop
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 4h data for HTF Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Calculate 12h volume average (20-period)
    vol_avg_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1  # Positive values
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.zeros_like(values)
        result[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI values
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    dx = np.where((plus_di14 + minus_di14) == 0, 0, dx)
    adx = wilders_smoothing(dx, 14)
    
    # Align all HTF indicators to 4h primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h, additional_delay_bars=1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x 20-period average
        # Get the 12h bar index for current 4h bar (each 12h bar = 3 4h bars)
        idx_12h = i // 3
        if idx_12h >= len(volume_12h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_12h[idx_12h] > 1.8 * vol_avg_20_12h_aligned[i]
        
        # Breakout conditions
        breakout_long = close[i] > donchian_high_aligned[i]  # Price above upper Donchian
        breakout_short = close[i] < donchian_low_aligned[i]  # Price below lower Donchian
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx_aligned[i] > 25
        
        # Entry conditions
        enter_long = breakout_long and volume_confirmed and trend_filter
        enter_short = breakout_short and volume_confirmed and trend_filter
        
        # Exit conditions: opposite Donchian breakout or ADX < 20 (choppy market)
        exit_long = position == 1 and (close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20)
        exit_short = position == -1 and (close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0