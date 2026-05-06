#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h ADX trend filter with 4h Bollinger Band breakout and volume confirmation
# Long when price breaks above 4h BB upper band with volume > 1.5x 20-period average and 12h ADX > 25
# Short when price breaks below 4h BB lower band with volume > 1.5x 20-period average and 12h ADX > 25
# Uses Bollinger Bands for volatility-based breakouts, ADX for trend strength confirmation, volume for conviction
# Designed to capture strong moves in both bull and bear markets by filtering for trending conditions
# Target: 20-40 trades per year (80-160 over 4 years) with 0.25 position sizing

name = "4h_BollingerBreakout_ADX25_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Bollinger Bands (20-period, 2 std dev)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Calculate 12h ADX (14-period) for trend strength
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM (14-period)
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI and DX
    plus_di_14 = 100 * (plus_dm_14 / tr_14)
    minus_di_14 = 100 * (minus_dm_14 / tr_14)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    
    # Calculate ADX (14-period smoothed DX)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Bollinger warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above BB upper with volume confirmation and ADX > 25
            if close[i] > bb_upper[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below BB lower with volume confirmation and ADX > 25
            elif close[i] < bb_lower[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below BB lower band (mean reversion)
            if close[i] < bb_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above BB upper band (mean reversion)
            if close[i] > bb_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals