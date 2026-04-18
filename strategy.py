#!/usr/bin/env python3
"""
4h_Adaptive_Breakout_Range_V1
Hypothesis: Trade breakouts from Donchian channels (20) only when volatility is low (ATR ratio < 0.8) and volume confirms (>1.5x average), otherwise fade the breakout in ranging markets (ADX < 25). Uses 1d ADX for regime filter and 4h ATR for volatility filter. Designed to avoid false breakouts in chop and capture real trends. Works in bull via trend-following breakouts and in bear via mean-reversion fades. Target: 20-40 trades/year by combining volatility and regime filters to reduce false signals.
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
    
    # Get 4h data for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    donch_len = 20
    upper_4h = np.full_like(high_4h, np.nan)
    lower_4h = np.full_like(low_4h, np.nan)
    
    if len(high_4h) >= donch_len:
        for i in range(donch_len, len(high_4h)):
            upper_4h[i] = np.max(high_4h[i-donch_len:i])
            lower_4h[i] = np.min(low_4h[i-donch_len:i])
    
    # 4h ATR (14-period) for volatility filter
    atr_period = 14
    tr = np.maximum(np.maximum(high_4h[1:] - low_4h[1:], np.abs(high_4h[1:] - close_4h[:-1])), np.abs(low_4h[1:] - close_4h[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr_4h = np.full_like(close_4h, np.nan)
    
    if len(tr) >= atr_period:
        for i in range(atr_period, len(tr)):
            atr_4h[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Get 1d data for ADX (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = np.abs(np.diff(high))
        tr2 = np.abs(np.diff(low))
        tr3 = np.abs(np.subtract(high[1:], low[:-1]))
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])
        
        # Directional Movement
        up_move = np.diff(high)
        down_move = -np.diff(low)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = np.full_like(tr, np.nan)
        plus_di = np.full_like(tr, np.nan)
        minus_di = np.full_like(tr, np.nan)
        
        if len(tr) >= period:
            # Initial average
            atr[period] = np.mean(tr[1:period+1])
            plus_dm_sum = np.sum(plus_dm[1:period+1])
            minus_dm_sum = np.sum(minus_dm[1:period+1])
            
            # Wilder smoothing
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                plus_dm_sum = plus_dm_sum * (period - 1) / period + plus_dm[i]
                minus_dm_sum = minus_dm_sum * (period - 1) / period + minus_dm[i]
                plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
                minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
            
            # DX and ADX
            dx = np.full_like(tr, np.nan)
            adx = np.full_like(tr, np.nan)
            
            for i in range(period*2, len(tr)):
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            
            if len(tr) >= period*2 + 1:
                adx[period*2] = np.mean(dx[period+1:period*2+1])
                for i in range(period*2+1, len(tr)):
                    adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to 4h timeframe
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_len, atr_period, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR ratio < 0.8 (low volatility)
        atr_ratio = atr_4h_aligned[i] / np.mean(atr_4h_aligned[max(0, i-50):i+1]) if i >= 50 else 1.0
        low_vol = atr_ratio < 0.8
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filter: ADX < 25 = ranging, ADX > 25 = trending
        ranging = adx_1d_aligned[i] < 25
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # In ranging market: fade breakouts (mean reversion)
            if ranging and vol_confirm:
                if close[i] > upper_4h_aligned[i]:
                    signals[i] = -0.25  # short breakout fade
                    position = -1
                elif close[i] < lower_4h_aligned[i]:
                    signals[i] = 0.25   # long breakout fade
                    position = 1
            # In trending market: follow breakouts
            elif trending and low_vol and vol_confirm:
                if close[i] > upper_4h_aligned[i]:
                    signals[i] = 0.25   # long breakout
                    position = 1
                elif close[i] < lower_4h_aligned[i]:
                    signals[i] = -0.25  # short breakout
                    position = -1
        
        elif position == 1:
            # Long exit: price re-enters Donchian channel or volatility spikes
            if close[i] < upper_4h_aligned[i] or atr_ratio > 1.2:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price re-enters Donchian channel or volatility spikes
            if close[i] > lower_4h_aligned[i] or atr_ratio > 1.2:
                signals[i] = 0.25   # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Adaptive_Breakout_Range_V1"
timeframe = "4h"
leverage = 1.0