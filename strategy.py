#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with 12h/1d Donchian breakout + volume confirmation + regime filter.
# Long when price breaks above 12h Donchian(20) high with volume > 1.5x 12h average volume and 12h ADX > 25.
# Short when price breaks below 12h Donchian(20) low with volume > 1.5x and 12h ADX > 25.
# Exit when price crosses 12h Donchian midline (average of high/low channel) or ADX < 20.
# Designed for low trade frequency (~15-30/year) to minimize fee decay while capturing strong trends.
# Works in bull/bear markets by only trading when ADX indicates strong trend (ADX>25) and avoiding chop.

name = "6h_12h_donchian_breakout_volume_adx_v1"
timeframe = "6h"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian high and low
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 12h ADX (14-period)
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def _wilders_smoothing(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.nansum(arr[:period])  # First value is simple average
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr = _wilders_smoothing(tr, 14)
    plus_di = 100 * _wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * _wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = _wilders_smoothing(dx, 14)
    
    # 12h average volume (20-period)
    vol_12h = df_12h['volume'].values
    vol_avg_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 20 to ensure all indicators are valid
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.5 * 12h average volume
        vol_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        # Trend filter: 12h ADX > 25 (strong trend)
        trend_filter = adx_aligned[i] > 25
        
        # Exit trend filter: ADX < 20 (weakening trend)
        exit_trend_filter = adx_aligned[i] < 20
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        long_breakout = high[i] > donch_high_aligned[i]
        short_breakout = low[i] < donch_low_aligned[i]
        
        long_entry = long_breakout and vol_filter and trend_filter
        short_entry = short_breakout and vol_filter and trend_filter
        
        # Exit conditions: midline cross or trend weakening
        exit_long = (close[i] < donch_mid_aligned[i]) or exit_trend_filter
        exit_short = (close[i] > donch_mid_aligned[i]) or exit_trend_filter
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
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