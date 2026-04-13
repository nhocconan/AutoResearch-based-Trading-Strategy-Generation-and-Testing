#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX regime filter.
    # Long when price breaks above Donchian upper band + volume > 1.5x 20-period avg + ADX_1d > 20.
    # Short when price breaks below Donchian lower band + volume > 1.5x 20-period avg + ADX_1d > 20.
    # Exit when price crosses Donchian midpoint (mean reversion in ranging markets).
    # Uses Donchian for structure, volume for conviction, ADX to filter ranging markets.
    # Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Get 1d data for volume and ADX (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate volume moving average (20-period) on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate True Range (TR) on 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate +DI and -DI (14) on 1d
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Wilder's smoothing function
    def wilders_smoothing(values, period):
        alpha = 1.0 / period
        smoothed = np.zeros_like(values)
        if len(values) >= period:
            smoothed[period-1] = np.mean(values[:period])  # First value is simple average
            for i in range(period, len(values)):
                smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, 14) / np.maximum(atr_1d, 1e-10)
    minus_di_1d = 100 * wilders_smoothing(minus_dm, 14) / np.maximum(atr_1d, 1e-10)
    
    # Calculate ADX (14) on 1d
    dx = 100 * np.abs(plus_di_1d - minus_di_1d) / np.maximum(plus_di_1d + minus_di_1d, 1e-10)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Align HTF indicators to 4h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Regime filter: ADX_1d > 20 indicates sufficient trend strength
        regime_filter = adx_1d_aligned[i] > 20
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions: price crosses Donchian midpoint
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and volume_spike and regime_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and volume_spike and regime_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "4h_1d_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0