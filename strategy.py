#!/usr/bin/env python3

"""
Hypothesis: 4-hour RSI(2) Mean Reversion with 1-day ADX Trend Filter and Volume Confirmation.
Trades extreme RSI(2) readings (<10 for long, >90 for short) only when the daily trend is strong (ADX>25).
Uses volume spike to confirm institutional interest at extreme levels. Designed for low trade frequency
(15-30 trades/year) to minimize fee decay and work in both bull and bear markets by only trading
with the higher timeframe trend during overextended moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    atr[period-1] = np.mean(tr[:period])
    dm_plus_smooth[period-1] = np.mean(dm_plus[:period])
    dm_minus_smooth[period-1] = np.mean(dm_minus[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
    
    # Calculate DI+ and DI-
    plus_di = 100 * dm_plus_smooth / atr
    minus_di = 100 * dm_minus_smooth / atr
    
    # Calculate DX and ADX
    dx = np.zeros_like(close)
    dx[period:] = 100 * np.abs(plus_di[period:] - minus_di[period:]) / (plus_di[period:] + minus_di[period:])
    
    adx = np.zeros_like(close)
    adx[2*period-1:] = np.mean(dx[period-1:2*period-1])  # First ADX value
    for i in range(2*period, len(close)):
        adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def calculate_rsi(close, period=2):
    """Calculate RSI"""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # Same length as close
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period-1] = np.mean(gain[:period])
    avg_loss[period-1] = np.mean(loss[:period])
    
    for i in range(period, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily ADX for trend filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # RSI(2) on 4h close
    rsi_2 = calculate_rsi(close, 2)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[19:] = np.convolve(volume, np.ones(20)/20, mode='valid')
    vol_ma_20 = np.concatenate([np.full(19, np.nan), vol_ma_20[19:]])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(adx_14_1d_aligned[i]) or np.isnan(rsi_2[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: only trade with strong daily trend
        strong_trend = adx_14_1d_aligned[i] > 25
        
        if position == 0 and vol_spike and strong_trend:
            # Long: extreme oversold RSI(2) < 10
            if rsi_2[i] < 10:
                signals[i] = 0.25
                position = 1
            # Short: extreme overbought RSI(2) > 90
            elif rsi_2[i] > 90:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: RSI returns to neutral zone (40-60)
            exit_signal = False
            
            if position == 1:
                # Exit long: RSI crosses above 40
                if rsi_2[i] > 40:
                    exit_signal = True
            else:  # position == -1
                # Exit short: RSI crosses below 60
                if rsi_2[i] < 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_RSI2_MeanReversion_1dADX25_Volume"
timeframe = "4h"
leverage = 1.0