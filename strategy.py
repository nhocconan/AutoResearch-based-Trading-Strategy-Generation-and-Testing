#!/usr/bin/env python3
"""
4h_Williams_Alligator_Trend_v1
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength. 
Long when Lips > Teeth > Jaw (bullish alignment) with volume confirmation.
Short when Lips < Teeth < Jaw (bearish alignment) with volume confirmation.
Exit when alignment breaks or volume weakens. Uses 1d ADX filter to avoid weak trends.
Target: 20-40 trades/year by requiring strong alignment and volume confirmation.
Works in trending markets via trend following and avoids whipsaws in ranging markets.
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
    
    # Get daily data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator (SMMA-based)
    def smoothed_moving_average(data, period):
        """Williams Alligator uses SMMA (Smoothed Moving Average)"""
        if len(data) < period:
            return np.full_like(data, np.nan)
        
        sma = np.full_like(data, np.nan)
        smma = np.full_like(data, np.nan)
        
        # Calculate initial SMA
        sma[period-1] = np.mean(data[:period])
        smma[period-1] = sma[period-1]
        
        # Calculate SMMA using Wilder's smoothing
        for i in range(period, len(data)):
            smma[i] = (smma[i-1] * (period-1) + data[i]) / period
        
        return smma
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smoothed_moving_average(close, 13)
    teeth = smoothed_moving_average(close, 8)
    lips = smoothed_moving_average(close, 5)
    
    # Calculate 14-period ADX for trend strength filter
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
    
    # Align Alligator lines and ADX to 4h timeframe
    jaw_4h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5, 14) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or 
            np.isnan(lips_4h[i]) or np.isnan(adx_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_4h[i] > 25
        
        # Alligator alignment signals
        bullish_alignment = lips_4h[i] > teeth_4h[i] and teeth_4h[i] > jaw_4h[i]
        bearish_alignment = lips_4h[i] < teeth_4h[i] and teeth_4h[i] < jaw_4h[i]
        
        if position == 0:
            # Long: bullish alignment with volume and strong trend
            if bullish_alignment and vol_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment with volume and strong trend
            elif bearish_alignment and vol_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: alignment breaks or trend weakens
            if not bullish_alignment or not strong_trend:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: alignment breaks or trend weakens
            if not bearish_alignment or not strong_trend:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_Trend_v1"
timeframe = "4h"
leverage = 1.0