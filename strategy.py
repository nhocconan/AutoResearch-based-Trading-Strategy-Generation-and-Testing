#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 1d ADX trend filter
# - Entry: Long when price breaks above 20-period 4h high + 12h volume > 2.0x 20-period average + 1d ADX > 25 (trending market)
#          Short when price breaks below 20-period 4h low + 12h volume > 2.0x 20-period average + 1d ADX > 25 (trending market)
# - Exit: Close-based reversal - exit long when price < 4h mid-point, exit short when price > 4h mid-point
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Donchian channels for structure, volume for confirmation, 1d ADX for trend filter
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within HARD MAX: 400 total
# - Designed for 4h timeframe with volume confirmation and trend filter to reduce false breakouts
# - Works in both bull and bear markets by requiring ADX > 25 (trending condition) for entries

name = "4h_12h_1d_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Pre-compute 12h volume for confirmation
    volume_12h = df_12h['volume'].values
    
    # Pre-compute 1d OHLC for ADX calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_ma_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_4h = (high_ma_20_4h + low_ma_20_4h) / 2.0
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
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
    high_donch_4h_aligned = align_htf_to_ltf(prices, prices, high_ma_20_4h)  # 4h data already aligned
    low_donch_4h_aligned = align_htf_to_ltf(prices, prices, low_ma_20_4h)
    mid_4h_aligned = align_htf_to_ltf(prices, prices, mid_4h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_donch_4h_aligned[i]) or np.isnan(low_donch_4h_aligned[i]) or 
            np.isnan(mid_4h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h close
        close_price = close_4h[i]
        
        # Get current 12h volume for confirmation (need to align raw volume)
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirmation = volume_12h_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trend_filter = adx_1d_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 20-period high + volume confirmation + trending market
            if (close_price > high_donch_4h_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low + volume confirmation + trending market
            elif (close_price < low_donch_4h_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < 4h mid-point
            # Exit short when price > 4h mid-point
            if position == 1:
                if close_price < mid_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > mid_4h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals