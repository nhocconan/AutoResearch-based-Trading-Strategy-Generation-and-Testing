#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ADX(14) filter and volume confirmation.
# Long when price breaks above Donchian upper band, ADX > 25, and volume > 1.5x average.
# Short when price breaks below Donchian lower band, ADX > 25, and volume > 1.5x average.
# Exit when price reverses to touch the opposite Donchian band or ADX drops below 20.
# Designed for 15-25 trades/year on 12h timeframe with strong trend capture and low turnover.
# Uses ADX to filter only strong trends, avoiding whipsaws in ranging markets.
# Volume filter ensures breakouts have institutional participation.

name = "12h_1d_donchian_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    def _wilder_smooth(values, period):
        smoothed = np.zeros_like(values)
        smoothed[period-1] = np.nansum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    tr_sum = _wilder_smooth(tr, 14)
    plus_dm_sum = _wilder_smooth(plus_dm, 14)
    minus_dm_sum = _wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(tr_sum > 0, 100 * plus_dm_sum / tr_sum, 0)
    minus_di = np.where(tr_sum > 0, 100 * minus_dm_sum / tr_sum, 0)
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = _wilder_smooth(dx, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_window, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_14_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 20-period average volume
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: ADX > 25
        strong_trend = adx_14_1d_aligned[i] > 25
        
        # Weak trend filter for exit: ADX < 20
        weak_trend = adx_14_1d_aligned[i] < 20
        
        # Entry conditions
        bullish_breakout = (high[i] > donchian_high[i-1]) and vol_filter and strong_trend
        bearish_breakout = (low[i] < donchian_low[i-1]) and vol_filter and strong_trend
        
        # Exit conditions
        exit_long = (low[i] < donchian_low[i]) or weak_trend
        exit_short = (high[i] > donchian_high[i]) or weak_trend
        
        # Priority: entry > exit > hold
        if bullish_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals