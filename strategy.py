#!/usr/bin/env python3
"""
6h_AdaptiveKeltner_RSI2_Trend
Hypothesis: Combine RSI(2) mean reversion with Keltner Channel (ATR-based) breakouts on 6h,
filtered by 1d ADX trend strength. In high ADX (>25), trade breakouts; in low ADX (<20),
trade RSI(2) reversals. Uses volatility-adjusted position sizing to manage risk.
Works in bull (breakouts) and bear (mean reversion in ranges). Target: 50-150 total trades.
"""

name = "6h_AdaptiveKeltner_RSI2_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        if n < period * 2:
            return np.full(n, np.nan)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        up_move = high - np.roll(high, 1)
        down_move = np.roll(low, 1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)
        
        # Initial values
        atr[period-1] = np.mean(tr[:period])
        plus_dm_smooth[period-1] = np.mean(plus_dm[:period])
        minus_dm_smooth[period-1] = np.mean(minus_dm[:period])
        
        # Wilder's smoothing
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Directional Indicators
        plus_di = np.where(atr != 0, plus_dm_smooth / atr * 100, 0)
        minus_di = np.where(atr != 0, minus_dm_smooth / atr * 100, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = np.full(n, np.nan)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # 6h Keltner Channel (ATR-based)
    def calculate_atr(high, low, close, period):
        n = len(high)
        if n < period:
            return np.full(n, np.nan)
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.zeros(n)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_10 = calculate_atr(high, low, close, 10)
    atr_ma_20 = np.full(n, np.nan)
    if n >= 20:
        atr_ma_20[19] = np.mean(atr_10[:20])
        for i in range(20, n):
            atr_ma_20[i] = (atr_ma_20[i-1] * 19 + atr_10[i]) / 20
    
    # Keltner Channels
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    kc_upper = ema20 + 2 * atr_ma_20
    kc_lower = ema20 - 2 * atr_ma_20
    
    # RSI(2) for mean reversion
    def calculate_rsi(close, period):
        n = len(close)
        if n < period + 1:
            return np.full(n, np.nan)
        
        delta = np.diff(close)
        delta = np.concatenate([[np.nan], delta])
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros(n)
        avg_loss = np.zeros(n)
        
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, n):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi2 = calculate_rsi(close, 2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(adx_14_1d_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or np.isnan(rsi2[i]) or np.isnan(ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility-adjusted size (inverse vol)
        vol_factor = np.clip(atr_10[i] / np.nanmedian(atr_10[i-50:i+1] if i >= 50 else atr_10[:i+1]), 0.5, 2.0)
        base_size = 0.25
        size = base_size / vol_factor
        size = np.clip(size, 0.15, 0.35)
        
        if position == 0:
            # High ADX (>25): trend mode - trade breakouts
            if adx_14_1d_aligned[i] > 25:
                if close[i] > kc_upper[i]:
                    signals[i] = size
                    position = 1
                elif close[i] < kc_lower[i]:
                    signals[i] = -size
                    position = -1
            # Low ADX (<20): range mode - trade RSI extremes
            elif adx_14_1d_aligned[i] < 20:
                if rsi2[i] < 10 and close[i] > ema20[i]:  # Oversold but above average
                    signals[i] = size
                    position = 1
                elif rsi2[i] > 90 and close[i] < ema20[i]:  # Overbought but below average
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit: reverse signal or volatility expansion
            if (adx_14_1d_aligned[i] < 20 and rsi2[i] > 70) or close[i] < kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: reverse signal or volatility expansion
            if (adx_14_1d_aligned[i] < 20 and rsi2[i] < 30) or close[i] > kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals