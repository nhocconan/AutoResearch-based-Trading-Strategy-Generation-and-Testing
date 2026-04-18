# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
[60578] 1d_KAMA_Direction_PriceAction_Regime_Filter
Hypothesis: On the daily timeframe, KAMA (Kaufman Adaptive Moving Average) adapts to market noise,
providing a responsive trend filter. Combine with price action (close vs KAMA) and a regime filter
using weekly ADX to avoid whipsaws in strong trends. Use volume confirmation to ensure conviction.
Designed for low trade frequency (10-25 trades/year) on 1d timeframe to minimize fee drag.
Works in bull markets by following KAMA trend; in bear markets, the regime filter prevents counter-trend trades.
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
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    def calculate_kama(close, volume, er_length=10, fast_sc=2, slow_sc=30):
        # Change and volatility
        change = np.abs(np.diff(close, prepend=close[0]))
        vol = np.sum(np.abs(np.diff(close, prepend=close[0]))[:er_length])  # placeholder, will adjust
        
        # Efficiency Ratio (ER)
        er = np.zeros_like(close)
        for i in range(er_length, len(close)):
            if np.sum(np.abs(np.diff(close[i-er_length:i+1]))) > 0:
                er[i] = np.abs(close[i] - close[i-er_length]) / np.sum(np.abs(np.diff(close[i-er_length:i+1])))
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        
        return kama
    
    # Calculate KAMA on daily data
    kama_1d = calculate_kama(close_1d, volume_1d, er_length=10, fast_sc=2, slow_sc=30)
    
    # Get weekly data for ADX (regime filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX on weekly data
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
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align KAMA and ADX to daily timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20) + 5  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: weekly ADX < 25 (avoid strong trends where counter-trend signals fail)
        not_strong_trend = adx_1w_aligned[i] < 25
        
        if position == 0:
            # Long: price above KAMA with volume confirmation and not strong trend
            if close[i] > kama_1d_aligned[i] and vol_confirm and not_strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume confirmation and not strong trend
            elif close[i] < kama_1d_aligned[i] and vol_confirm and not_strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below KAMA OR strong trend emerges (ADX > 30)
            if close[i] < kama_1d_aligned[i] or adx_1w_aligned[i] > 30:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above KAMA OR strong trend emerges (ADX > 30)
            if close[i] > kama_1d_aligned[i] or adx_1w_aligned[i] > 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_PriceAction_Regime_Filter"
timeframe = "1d"
leverage = 1.0