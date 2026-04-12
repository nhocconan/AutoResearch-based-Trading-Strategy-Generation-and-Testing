#!/usr/bin/env python3
"""
4h_1d_1w_MeanReversion_RSI_Extremes
Hypothesis: On 4h timeframe, enter mean-reversion trades at daily extreme RSI levels (RSI < 30 for long, RSI > 70 for short) only when weekly volatility regime is trending (ADX > 25) to avoid choppy markets. Uses RSI for overextension detection and ADX for regime filter. Designed for low trade frequency (<40/year) and high win rate in both bull and bear markets by fading overextended moves during trending conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_MeanReversion_RSI_Extremes"
timeframe = "4h"
leverage = 1.0

def rsi(close, period=14):
    """Calculate RSI with proper handling"""
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= period:
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period + 1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+ and DM-
    def smooth(x, period):
        result = np.zeros_like(x)
        if len(x) >= period:
            result[period] = np.sum(x[:period])
            for i in range(period + 1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i-1]
        return result
    
    tr_smooth = smooth(tr, period)
    dm_plus_smooth = smooth(dm_plus, period)
    dm_minus_smooth = smooth(dm_minus, period)
    
    # Directional Indicators
    di_plus = np.where(tr_smooth != 0, 100 * dm_plus_smooth / tr_smooth, 0)
    di_minus = np.where(tr_smooth != 0, 100 * dm_minus_smooth / tr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_result = smooth(dx, period)
    
    return adx_result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY RSI FOR OVEREXTENSION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    rsi_1d = rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === WEEKLY ADX FOR TREND REGIME ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    adx_1w = adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Regime: ADX > 25 = trending market
    trending_regime = adx_1w_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Mean reversion signals at RSI extremes
        rsi_long_signal = rsi_1d_aligned[i] < 30  # Oversold
        rsi_short_signal = rsi_1d_aligned[i] > 70  # Overbought
        
        # Exit when RSI returns to neutral zone
        rsi_exit_long = rsi_1d_aligned[i] > 50
        rsi_exit_short = rsi_1d_aligned[i] < 50
        
        # Only trade in trending regime to avoid whipsaws in chop
        if trending_regime[i]:
            if rsi_long_signal and position != 1:
                position = 1
                signals[i] = 0.25
            elif rsi_short_signal and position != -1:
                position = -1
                signals[i] = -0.25
            elif rsi_exit_long and position == 1:
                position = 0
                signals[i] = 0.0
            elif rsi_exit_short and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # In ranging market, stay flat or exit positions
            if position == 1 and rsi_exit_long:
                position = 0
                signals[i] = 0.0
            elif position == -1 and rsi_exit_short:
                position = 0
                signals[i] = 0.0
            else:
                # Hold or close positions in ranging market
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals