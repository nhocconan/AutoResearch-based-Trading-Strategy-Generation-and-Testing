#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d ADX trend filter and volume spike confirmation
# Camarilla pivot levels identify intraday support/resistance where price often reverses or breaks out.
# 1d ADX > 25 ensures we only trade in trending markets, reducing whipsaw in ranging conditions.
# Volume spike (>1.5x 20 EMA) confirms institutional participation. Discrete sizing 0.25 limits risk.
# Works in bull/bear: trend filter prevents counter-trend entries. Target: 50-150 trades over 4 years.

name = "12h_Camarilla_R3S3_1dADX_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = pd.Series(df_1d['high'])
    low_1d = pd.Series(df_1d['low'])
    close_1d = pd.Series(df_1d['close'])
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d.diff()
    down_move = low_1d.diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=high_1d.index)
    minus_dm = pd.Series(minus_dm, index=high_1d.index)
    
    # Directional Indicators
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr)
    
    # ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    
    # Align 1d ADX to 12h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate Camarilla levels for 12h timeframe using previous day's OHLC
    # We need to calculate daily OHLC from 12h data, then compute Camarilla
    # For simplicity, we'll use rolling window to approximate daily OHLC
    # In practice, we should resample to 1d, but we avoid resampling per rules
    # Instead, we use the 1d data we already loaded for OHLC
    # Camarilla levels: based on previous day's range
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # We'll use the 1d data to compute these levels for each 12h bar
    
    # Since we have df_1d with actual 1d OHLC, we compute Camarilla from 1d
    # and align to 12h
    prev_close_1d = df_1d['close'].shift(1)
    prev_high_1d = df_1d['high'].shift(1)
    prev_low_1d = df_1d['low'].shift(1)
    
    camarilla_r3_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_s3_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d.values)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + trending + volume spike
            if close[i] > r3_aligned[i] and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + trending + volume spike
            elif close[i] < s3_aligned[i] and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla midpoint OR trend weakens OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] < midpoint or 
                adx_aligned[i] < 20 or  # trend weakening
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla midpoint OR trend weakens OR volume drops
            midpoint = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if (close[i] > midpoint or 
                adx_aligned[i] < 20 or  # trend weakening
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals