#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Donchian channels (20-period)
    donchian_high_1d = np.full(len(df_1d), np.nan)
    donchian_low_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        donchian_high_1d[i] = np.max(high_1d[i-20:i])
        donchian_low_1d[i] = np.min(low_1d[i-20:i])
    
    # Calculate daily ADX (14-period) for trend strength
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    plus_dm[0] = minus_dm[0] = np.nan
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values (14-period)
    def smooth_series(data, period):
        smoothed = np.full_like(data, np.nan)
        if len(data) < period:
            return smoothed
        # First value: simple average
        smoothed[period-1] = np.mean(data[1:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            smoothed[i] = (smoothed[i-1] * (period-1) + data[i]) / period
        return smoothed
    
    atr_1d = smooth_series(tr, 14)
    plus_di_1d = 100 * smooth_series(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * smooth_series(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = smooth_series(dx_1d, 14)
    
    # Calculate daily RSI (14-period) for overbought/oversold
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily indicators to 6h timeframe
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_1d_aligned[i]
        short_breakout = close[i] < donchian_low_1d_aligned[i]
        
        # RSI filter: avoid extreme overbought/oversold
        rsi_not_extreme = (rsi_1d_aligned[i] > 30) and (rsi_1d_aligned[i] < 70)
        
        # Entry conditions
        long_entry = long_breakout and strong_trend and rsi_not_extreme
        short_entry = short_breakout and strong_trend and rsi_not_extreme
        
        # Exit conditions: opposite breakout or loss of trend
        long_exit = short_breakout or (adx_1d_aligned[i] < 20)
        short_exit = long_breakout or (adx_1d_aligned[i] < 20)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_adx_donchian_rsi_filter_v1"
timeframe = "6h"
leverage = 1.0