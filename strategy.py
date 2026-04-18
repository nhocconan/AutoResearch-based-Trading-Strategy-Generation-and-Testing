#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum breakout with 4h/1d trend filter, volume confirmation, and session filter.
# Uses 4h EMA trend (bullish: price > EMA50, bearish: price < EMA50) and 1d ADX > 20 for trend strength.
# Entry on 1h: price breaks above/below 4h Donchian channel (20-period) with volume > 1.5x 20-period average.
# Exit when price crosses 4h EMA21 or volume drops below average.
# Session filter: only trade 08-20 UTC to avoid low-volume periods.
# Position size: 0.20 (20% of capital) to manage risk.
# Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
# Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns in downtrend).

name = "1h_EMA50_ADX_Donchian_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA and Donchian channel
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 and EMA21
    close_4h = pd.Series(df_4h['close'].values)
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_4h = close_4h.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14-period)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = abs(high_1d - close_1d.shift(1))
    tr3 = abs(low_1d - close_1d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = high_1d.diff()
    down_move = low_1d.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1d)
    
    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = dx.ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 1h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume filter: 1.5x 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Trend conditions
        # Bullish trend: price > 4h EMA50 AND ADX > 20
        bullish_trend = close[i] > ema50_4h_aligned[i] and adx_1d_aligned[i] > 20
        # Bearish trend: price < 4h EMA50 AND ADX > 20
        bearish_trend = close[i] < ema50_4h_aligned[i] and adx_1d_aligned[i] > 20
        
        if position == 0:
            # Long entry: bullish trend + price breaks above 4h Donchian high + volume filter
            if bullish_trend and close[i] > donchian_high_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: bearish trend + price breaks below 4h Donchian low + volume filter
            elif bearish_trend and close[i] < donchian_low_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA21 OR trend weakens (ADX < 15) OR volume drops
            if close[i] < ema21_4h_aligned[i] or adx_1d_aligned[i] < 15 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA21 OR trend weakens (ADX < 15) OR volume drops
            if close[i] > ema21_4h_aligned[i] or adx_1d_aligned[i] < 15 or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals