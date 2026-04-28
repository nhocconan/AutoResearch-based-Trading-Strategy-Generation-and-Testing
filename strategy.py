#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX25 regime filter and volume spike confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13. Measures bull/bear strength relative to trend.
# ADX > 25 indicates strong trend (use Elder Ray for continuation); ADX < 25 indicates ranging (fade extremes).
# Volume spike (>2.0x 20-bar average) confirms institutional participation.
# Position size 0.25 balances return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in bull markets via trend continuation and bear markets via mean reversion in ranges.
# Targets BTC and ETH primarily.

name = "6h_ElderRay_1dADX25_Regime_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Elder Ray calculation and 1d data for ADX regime filter
    df_6h = get_htf_data(prices, '6h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_6h) < 14 or len(df_1d) < 25:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power = high_6h - ema_13_6h  # Bull Power: High - EMA
    bear_power = low_6h - ema_13_6h   # Bear Power: Low - EMA
    
    # Calculate 1d ADX (14-period) for regime filter
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(close_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and DI
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 25  # Ensure sufficient history for ADX
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d ADX > 25 = trending, < 25 = ranging
        is_trending = adx_aligned[i] > 25
        is_ranging = adx_aligned[i] < 25
        
        # Elder Ray signals
        bull_strong = bull_power_aligned[i] > 0
        bear_strong = bear_power_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trending regime: Elder Ray continuation
        long_entry_trend = is_trending and bull_strong and vol_confirm
        short_entry_trend = is_trending and bear_strong and vol_confirm
        
        # Ranging regime: Elder Ray mean reversion (fade extremes)
        long_entry_range = is_ranging and (bear_power_aligned[i] < -np.std(bear_power_aligned[max(0,i-50):i+1])) and vol_confirm
        short_entry_range = is_ranging and (bull_power_aligned[i] > np.std(bull_power_aligned[max(0,i-50):i+1])) and vol_confirm
        
        long_entry = long_entry_trend or long_entry_range
        short_entry = short_entry_trend or short_entry_range
        
        # Exit conditions: opposing Elder Ray signal or regime change
        long_exit = bear_power_aligned[i] > 0 or (is_trending and not is_trending)  # Bear power positive or regime shift
        short_exit = bull_power_aligned[i] < 0 or (is_trending and not is_trending)  # Bull power negative or regime shift
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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