#!/usr/bin/env python3
"""
12h_KAMA_RSI_Trend_Breakout_v1
Hypothesis: Kaufman Adaptive Moving Average (KAMA) captures trend direction with low lag.
Long when price crosses above KAMA(10) with RSI(14) > 50 and volume confirmation.
Short when price crosses below KAMA(10) with RSI(14) < 50 and volume confirmation.
Uses 1d ADX < 25 to filter for range markets where mean reversion at KAMA works best.
Target: 15-25 trades/year by requiring trend alignment, momentum filter, and volume spike.
Works in bull/bear markets by following adaptive trend and avoiding whipsaws in low volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA(10) - close price only
    def calculate_kama(close, length=10):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=length))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.zeros_like(close)
        er[length:] = change / (volatility + 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[:] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, 10)
    
    # Calculate RSI(14)
    def calculate_rsi(close, length=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # Wilder's smoothing
        if len(close) > length:
            avg_gain[length] = np.mean(gain[:length])
            avg_loss[length] = np.mean(loss[:length])
            for i in range(length+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i-1]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i-1]) / length
        
        rs = np.zeros_like(close)
        rs[length:] = avg_gain[length:] / (avg_loss[length:] + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Get daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) for regime filter
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[np.nan], dm_plus])
        dm_minus = np.concatenate([[np.nan], dm_minus])
        
        # Smooth TR, DM+
        atr = np.full_like(tr, np.nan)
        dm_plus_smooth = np.full_like(dm_plus, np.nan)
        dm_minus_smooth = np.full_like(dm_minus, np.nan)
        
        if len(tr) >= period:
            # Initial values
            atr[period] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # DI+ and DI-
        di_plus = np.full_like(dm_plus_smooth, np.nan)
        di_minus = np.full_like(dm_minus_smooth, np.nan)
        valid = ~np.isnan(atr) & (atr != 0)
        di_plus[valid] = 100 * dm_plus_smooth[valid] / atr[valid]
        di_minus[valid] = 100 * dm_minus_smooth[valid] / atr[valid]
        
        # DX and ADX
        dx = np.full_like(di_plus, np.nan)
        dx_valid = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
        dx[dx_valid] = 100 * np.abs(di_plus[dx_valid] - di_minus[dx_valid]) / (di_plus[dx_valid] + di_minus[dx_valid])
        
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            # Initial ADX
            adx[2*period-1] = np.nanmean(dx[period:2*period])
            # Wilder smoothing for ADX
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d data to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 10) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: daily ADX < 25 (range/low volatility)
        low_vol = adx_12h[i] < 25
        
        if position == 0:
            # Long: price crosses above KAMA with RSI > 50 and volume in low volatility
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and rsi[i] > 50 and vol_confirm and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA with RSI < 50 and volume in low volatility
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and rsi[i] < 50 and vol_confirm and low_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA OR RSI < 40 (momentum loss)
            if close[i] < kama[i] and close[i-1] >= kama[i-1] or rsi[i] < 40:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA OR RSI > 60 (momentum loss)
            if close[i] > kama[i] and close[i-1] <= kama[i-1] or rsi[i] > 60:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_RSI_Trend_Breakout_v1"
timeframe = "12h"
leverage = 1.0