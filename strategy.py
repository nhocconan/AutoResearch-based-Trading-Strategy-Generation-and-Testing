#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX25 trend filter and volume confirmation.
# Uses proven Donchian structure with HTF ADX for regime filtering to avoid whipsaws in ranging markets.
# Long when price breaks above Donchian(20) high with volume and 12h ADX > 25 (trending).
# Short when price breaks below Donchian(20) low with volume and 12h ADX > 25 (trending).
# Volume spike (>2.0x 20-bar average) confirms breakout strength.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Works in both bull and bear via 12h ADX25 trend filter - only trades in trending regimes.

name = "6h_Donchian20_12hADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilder_smoothing(data, alpha):
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if np.isnan(result[i-1]) if i > 0 else True:
                result[i] = data[i]
            else:
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    tr_smoothed = wilder_smoothing(tr, alpha)
    plus_dm_smoothed = wilder_smoothing(plus_dm, alpha)
    minus_dm_smoothed = wilder_smoothing(minus_dm, alpha)
    
    # Plus Directional Indicator (+DI) and Minus Directional Indicator (-DI)
    plus_di = 100 * plus_dm_smoothed / tr_smoothed
    minus_di = 100 * minus_dm_smoothed / tr_smoothed
    
    # Directional Index (DX) and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smoothing(dx, alpha)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_high[i] and volume_spike[i]
        short_breakout = close[i] < donchian_low[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or loss of trend
        long_exit = close[i] < donchian_low[i] or not is_trending
        short_exit = close[i] > donchian_high[i] or not is_trending
        
        # Handle entries and exits
        if long_breakout and is_trending and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and is_trending and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals