#!/usr/bin/env python3
"""
1h_Pullback_to_4h_EMA_with_Volume_and_Session
Hypothesis: In trending markets (defined by 4h EMA21), price pulls back to the 4h EMA on 1h timeframe, offering high-probability entries. We enter long when price touches or crosses above the 4h EMA21 with volume confirmation (>1.5x 20-period average) during active hours (08-20 UTC). Short when price touches or crosses below the 4h EMA21 with volume confirmation. Uses 1d ADX>25 to filter for trending conditions only, avoiding whipsaws in ranging markets. Targets 15-30 trades/year by requiring multiple confluence factors. Works in bull markets by buying EMA pullbacks in uptrends, and in bear markets by selling EMA pullbacks in downtrends.
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
    
    # Get 4h data for EMA21 (trend filter)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA21 on 4h close
    ema_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 21:
        ema_4h[20] = np.mean(close_4h[:21])  # simple average for first value
        multiplier = 2 / (21 + 1)
        for i in range(21, len(close_4h)):
            ema_4h[i] = close_4h[i] * multiplier + ema_4h[i-1] * (1 - multiplier)
    
    # Align 4h EMA21 to 1h timeframe (wait for bar close)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for ADX25 (trend strength filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
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
        
        # Smoothed values
        atr = np.full_like(high, np.nan)
        dm_plus_smooth = np.full_like(high, np.nan)
        dm_minus_smooth = np.full_like(high, np.nan)
        
        if len(high) >= period:
            # First values: simple average
            atr[period-1] = np.nanmean(tr[1:period+1])
            dm_plus_smooth[period-1] = np.nanmean(dm_plus[1:period+1])
            dm_minus_smooth[period-1] = np.nanmean(dm_minus[1:period+1])
            
            # Wilder smoothing
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
                dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period - 1) + dm_plus[i]) / period
                dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period - 1) + dm_minus[i]) / period
        
        # Directional Indicators
        di_plus = np.full_like(high, np.nan)
        di_minus = np.full_like(high, np.nan)
        dx = np.full_like(high, np.nan)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX: smoothed DX
        adx = np.full_like(high, np.nan)
        if len(high) >= 2 * period - 1:
            adx[2*period-2] = np.nanmean(dx[period:2*period])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d ADX to 1h timeframe (wait for bar close)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in trending markets (ADX > 25) and during session
        if not (adx_1d_aligned[i] > 25 and session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price crosses above 4h EMA21 with volume confirmation
            if (close[i] > ema_4h_aligned[i] and close[i-1] <= ema_4h_aligned[i-1] and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price crosses below 4h EMA21 with volume confirmation
            elif (close[i] < ema_4h_aligned[i] and close[i-1] >= ema_4h_aligned[i-1] and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA21 (trend change)
            if close[i] < ema_4h_aligned[i] and close[i-1] >= ema_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA21 (trend change)
            if close[i] > ema_4h_aligned[i] and close[i-1] <= ema_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pullback_to_4h_EMA_with_Volume_and_Session"
timeframe = "1h"
leverage = 1.0