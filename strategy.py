#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# - Entry: Long when price breaks above 20-period 12h high + 1d volume > 2.0x 20-period average + 1w ADX > 25 (trending market)
#          Short when price breaks below 20-period 12h low + 1d volume > 2.0x 20-period average + 1w ADX > 25 (trending market)
# - Exit: Close-based reversal - exit long when price < 12h mid-point, exit short when price > 12h mid-point
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Donchian channels for structure, volume for confirmation, 1w ADX for trend filter
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 200 total
# - Designed for 12h timeframe to reduce trade frequency and avoid fee drag, with HTF filters for robustness

name = "12h_1d_1w_donchian_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Pre-compute 1d volume for confirmation
    volume_1d = df_1d['volume'].values
    
    # Pre-compute 1w OHLC for ADX calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_ma_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_ma_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    mid_12h = (high_ma_20_12h + low_ma_20_12h) / 2.0
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w ADX (14-period)
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    atr_1w = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align all HTF data to 12h timeframe
    high_donch_12h_aligned = align_htf_to_ltf(prices, prices, high_ma_20_12h)  # 12h data already aligned
    low_donch_12h_aligned = align_htf_to_ltf(prices, prices, low_ma_20_12h)
    mid_12h_aligned = align_htf_to_ltf(prices, prices, mid_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(high_donch_12h_aligned[i]) or np.isnan(low_donch_12h_aligned[i]) or 
            np.isnan(mid_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get current 1d volume for confirmation (need to align raw volume)
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # Trend filter: 1w ADX > 25 indicates trending market
        trend_filter = adx_1w_aligned[i] > 25.0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 20-period high + volume confirmation + trending market
            if (close_price > high_donch_12h_aligned[i] and 
                volume_confirmation and 
                trend_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low + volume confirmation + trending market
            elif (close_price < low_donch_12h_aligned[i] and 
                  volume_confirmation and 
                  trend_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < 12h mid-point
            # Exit short when price > 12h mid-point
            if position == 1:
                if close_price < mid_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > mid_12h_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals