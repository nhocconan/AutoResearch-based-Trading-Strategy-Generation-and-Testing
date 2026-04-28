# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_Triple_RSI_Momentum_VolumeFilter
Hypothesis: Uses three RSI periods (short, medium, long) to capture momentum shifts.
RSI(6) < 30 and RSI(14) > RSI(50) signals bullish momentum; RSI(6) > 70 and RSI(14) < RSI(50) signals bearish momentum.
Requires volume confirmation (current volume > 20-period average) to avoid false signals.
Uses 1-day ADX > 25 as trend filter to ensure we trade in trending markets only.
Designed for low trade frequency (<30/year) on 12h timeframe to minimize fee drag while capturing sustained moves.
Works in both bull and bear markets by filtering for trend strength and momentum exhaustion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = np.zeros_like(tr)
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    
    # Initial smoothed values (simple average)
    atr[tr_period] = np.mean(tr[1:tr_period+1])
    dm_plus_smooth[tr_period] = np.mean(dm_plus[1:tr_period+1])
    dm_minus_smooth[tr_period] = np.mean(dm_minus[1:tr_period+1])
    
    # Wilder smoothing
    for i in range(tr_period + 1, len(tr)):
        atr[i] = (atr[i-1] * (tr_period - 1) + tr[i]) / tr_period
        dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (tr_period - 1) + dm_plus[i]) / tr_period
        dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (tr_period - 1) + dm_minus[i]) / tr_period
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = np.zeros_like(dx)
    adx[2*tr_period] = np.mean(dx[tr_period:2*tr_period])
    for i in range(2*tr_period + 1, len(dx)):
        adx[i] = (adx[i-1] * (tr_period - 1) + dx[i]) / tr_period
    
    # Align ADX to 12h timeframe (with 1-bar delay for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate multiple RSI on 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    def calculate_rsi(prices, period):
        rsi = np.zeros_like(prices)
        if len(prices) < period + 1:
            return rsi
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # First average gain/loss
        avg_gain = np.mean(gain[:period])
        avg_loss = np.mean(loss[:period])
        
        rsi[period] = 100 - (100 / (1 + (avg_gain / avg_loss if avg_loss != 0 else 0)))
        
        # Wilder smoothing
        for i in range(period + 1, len(prices)):
            avg_gain = (avg_gain * (period - 1) + gain[i-1]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i-1]) / period
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    rsi_6 = calculate_rsi(close_12h, 6)
    rsi_14 = calculate_rsi(close_12h, 14)
    rsi_50 = calculate_rsi(close_12h, 50)
    
    # Align RSI to main timeframe
    rsi_6_aligned = align_htf_to_ltf(prices, df_12h, rsi_6)
    rsi_14_aligned = align_htf_to_ltf(prices, df_12h, rsi_14)
    rsi_50_aligned = align_htf_to_ltf(prices, df_12h, rsi_50)
    
    # Volume confirmation: current volume > 20-period average
    volume_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        volume_ma[i] = np.mean(volume[i-20:i])
    volume_ma[:20] = volume_ma[20] if len(volume) > 20 else 0
    volume_confirm = volume > volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_6_aligned[i]) or
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(rsi_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # RSI momentum signals
        rsi_6 = rsi_6_aligned[i]
        rsi_14 = rsi_14_aligned[i]
        rsi_50 = rsi_50_aligned[i]
        
        # Bullish: short RSI oversold, medium RSI above long RSI (momentum building)
        bullish_momentum = (rsi_6 < 30) and (rsi_14 > rsi_50)
        
        # Bearish: short RSI overbought, medium RSI below long RSI (momentum fading)
        bearish_momentum = (rsi_6 > 70) and (rsi_14 < rsi_50)
        
        # Entry conditions
        long_entry = bullish_momentum and trending and volume_confirm[i]
        short_entry = bearish_momentum and trending and volume_confirm[i]
        
        # Exit conditions: momentum exhaustion or trend weakening
        long_exit = (rsi_6 > 50) or (adx_aligned[i] < 20)  # RSI recovery or trend weakening
        short_exit = (rsi_6 < 50) or (adx_aligned[i] < 20)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Triple_RSI_Momentum_VolumeFilter"
timeframe = "12h"
leverage = 1.0