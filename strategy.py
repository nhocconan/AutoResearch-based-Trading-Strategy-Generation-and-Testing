#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d ADX regime filter
# - Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
# - 1d ADX > 25 = trending regime (use Elder Ray for momentum entries)
# - 1d ADX <= 25 = ranging regime (fade extreme Elder Ray values)
# - In trending: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling
# - In ranging: long when Bear Power < -std(Bear Power) and turning up, short when Bull Power > std(Bull Power) and turning down
# - Volume confirmation: current 6h volume > 1.2x 20-period average
# - Fixed position size 0.25 to control drawdown
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years)
# - Works in both bull (trending regime captures momentum) and bear (ranging regime fades extremes)

name = "6h_1d_elder_ray_adx_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Calculate 1d ADX for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / np.where(atr == 0, 1, atr)
    minus_di = 100 * minus_dm_smooth / np.where(atr == 0, 1, atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all 1d data to 6h timeframe (wait for completed 1d bar)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute Elder Ray statistics for ranging regime (using aligned values)
    # Calculate rolling mean and std for Bear Power and Bull Power
    bull_mean = pd.Series(bull_power_1d_aligned).rolling(window=50, min_periods=50).mean().values
    bull_std = pd.Series(bull_power_1d_aligned).rolling(window=50, min_periods=50).std().values
    bear_mean = pd.Series(bear_power_1d_aligned).rolling(window=50, min_periods=50).mean().values
    bear_std = pd.Series(bear_power_1d_aligned).rolling(window=50, min_periods=50).std().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(bull_mean[i]) or np.isnan(bull_std[i]) or
            np.isnan(bear_mean[i]) or np.isnan(bear_std[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average
        volume_confirmed = volume[i] > 1.2 * vol_ma_20[i]
        
        # Regime filter: 1d ADX > 25 = trending, <= 25 = ranging
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] <= 25
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions
            if trending:
                # In trending: exit when Bull Power turns negative
                if bull_power_1d_aligned[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # ranging
                # In ranging: exit when Bear Power rises above -0.5 * std
                if bear_power_1d_aligned[i] > -0.5 * bear_std[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
                    
        elif position == -1:  # Short position
            # Exit conditions
            if trending:
                # In trending: exit when Bear Power turns positive
                if bear_power_1d_aligned[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # ranging
                # In ranging: exit when Bull Power falls below 0.5 * std
                if bull_power_1d_aligned[i] < 0.5 * bull_std[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
        else:  # Flat
            # Entry logic based on regime
            if volume_confirmed:
                if trending:
                    # In trending: momentum entries
                    # Long: Bull Power > 0 and rising (current > previous)
                    # Short: Bear Power < 0 and falling (current < previous)
                    if (bull_power_1d_aligned[i] > 0 and 
                        i > 100 and bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1]):
                        position = 1
                        signals[i] = position_size
                    elif (bear_power_1d_aligned[i] < 0 and 
                          i > 100 and bear_power_1d_aligned[i] < bear_power_1d_aligned[i-1]):
                        position = -1
                        signals[i] = -position_size
                else:  # ranging
                    # In ranging: mean reversion at extremes
                    # Long: Bear Power < -1.0 * std and turning up (current > previous)
                    # Short: Bull Power > 1.0 * std and turning down (current < previous)
                    if (bear_power_1d_aligned[i] < -1.0 * bear_std[i] and
                        i > 100 and bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1]):
                        position = 1
                        signals[i] = position_size
                    elif (bull_power_1d_aligned[i] > 1.0 * bull_std[i] and
                          i > 100 and bull_power_1d_aligned[i] < bull_power_1d_aligned[i-1]):
                        position = -1
                        signals[i] = -position_size
    
    return signals