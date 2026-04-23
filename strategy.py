#!/usr/bin/env python3
"""
Strategy: 1h RSI(14) mean reversion with 4h ADX(14) trend filter and volume confirmation
Long when RSI < 30 (oversold) + 4h ADX > 25 (trending) + volume > 1.3x 20-bar average
Short when RSI > 70 (overbought) + 4h ADX > 25 (trending) + volume > 1.3x 20-bar average
Exit when RSI crosses 50 (mean reversion midpoint)
Designed for low trade frequency (~15-35/year) to minimize fee drag in both bull and bear markets.
Works in trending markets (ADX filter) and avoids ranging markets where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data for ADX trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 4h data
    # ADX requires +DI, -DI, and TR calculation
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high_4h - np.roll(high_4h, 1)
    down_move = np.roll(low_4h, 1) - low_4h
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr_smoothed = wilders_smoothing(tr, 14)
    plus_dm_smoothed = wilders_smoothing(plus_dm, 14)
    minus_dm_smoothed = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(tr_smoothed != 0, 100 * plus_dm_smoothed / tr_smoothed, 0)
    minus_di = np.where(tr_smoothed != 0, 100 * minus_dm_smoothed / tr_smoothed, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 1h timeframe
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Calculate RSI (14-period) on 1h data
    def rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing for RSI
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        
        if len(gain) >= period:
            avg_gain[period-1] = np.mean(gain[:period])
            avg_loss[period-1] = np.mean(loss[:period])
            
            for i in range(period, len(gain)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_vals = rsi(close, 14)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(rsi_vals[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_4h_aligned[i]
        rsi_val = rsi_vals[i]
        rsi_prev = rsi_vals[i-1]
        vol_ratio = volume[i] / avg_volume[i] if avg_volume[i] > 0 else 0
        
        # Trend filter: only trade when ADX > 25 (trending market)
        is_trending = adx_val > 25
        
        # Volume confirmation: volume > 1.3x average
        volume_confirm = vol_ratio > 1.3
        
        if position == 0:
            # Long: RSI crosses above 30 (oversold recovery) + trending + volume
            if (rsi_val > 30 and rsi_prev <= 30 and 
                is_trending and volume_confirm):
                signals[i] = 0.20
                position = 1
            # Short: RSI crosses below 70 (overbought correction) + trending + volume
            elif (rsi_val < 70 and rsi_prev >= 70 and 
                  is_trending and volume_confirm):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: RSI crosses 50 (mean reversion midpoint)
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses below 50
                if rsi_val < 50 and rsi_prev >= 50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses above 50
                if rsi_val > 50 and rsi_prev <= 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_ADX_Trend_MeanReversion"
timeframe = "1h"
leverage = 1.0