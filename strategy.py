#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike (2.0x) and 1d ADX > 25 trend filter
# - Entry: Long when price breaks above 4h Donchian upper (20-period high) + 1d volume > 2.0x 20-period average + 1d ADX > 25
#          Short when price breaks below 4h Donchian lower (20-period low) + 1d volume > 2.0x 20-period average + 1d ADX > 25
# - Exit: Close-based reversal - exit long when price < 4h Donchian lower, exit short when price > 4h Donchian upper
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses 4h price channel for structure, 1d volume for confirmation, 1d ADX for trend filter
# - Target: 75-200 total trades over 4 years (19-50/year) to stay within HARD MAX: 400 total
# - Designed for 4h timeframe with strict volume confirmation (2.0x) and stronger trend filter (ADX>25) to reduce false breakouts
# - Works in both bull and bear markets by requiring ADX > 25 (strong trending condition) for entries

name = "4h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1d OHLC for ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_4h}), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_4h}), donchian_lower)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 1d volume for confirmation (need to align raw volume)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # Trend filter: 1d ADX > 25 indicates strong trending market
        trend_filter = adx_1d_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper + volume confirmation + strong trending market
            if (close_price > donchian_upper_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower + volume confirmation + strong trending market
            elif (close_price < donchian_lower_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < Donchian lower
            # Exit short when price > Donchian upper
            if position == 1:
                if close_price < donchian_lower_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > donchian_upper_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals