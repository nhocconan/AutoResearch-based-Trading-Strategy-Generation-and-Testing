#!/usr/bin/env python3
"""
Hypothesis: On the daily timeframe, price tends to respect weekly Ichimoku Cloud boundaries as dynamic support/resistance.
In trending markets (ADX > 25), breakouts above/below the cloud with volume confirmation (>1.5x 20-day average) capture strong moves.
In ranging markets (ADX <= 25), mean reversion at cloud edges (Tenkan/Kijun) with volume exhaustion provides counter-trend entries.
The Ichimoku Cloud provides inherent trend, momentum, and support/resistance in one indicator, reducing the need for multiple filters.
Designed for low trade frequency (10-25/year) to minimize fee drag while capturing major moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components: Tenkan, Kijun, Senkou Span A/B, Chikou"""
    n1 = 9   # Tenkan-sen period
    n2 = 26  # Kijun-sen period
    n3 = 52  # Senkou Span B period
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan = np.full_like(high, np.nan)
    for i in range(n1-1, len(high)):
        tenkan[i] = (np.max(high[i-n1+1:i+1]) + np.min(low[i-n1+1:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun = np.full_like(high, np.nan)
    for i in range(n2-1, len(high)):
        kijun[i] = (np.max(high[i-n2+1:i+1]) + np.min(low[i-n2+1:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i + n2 < len(high) and not np.isnan(tenkan[i]) and not np.isnan(kijun[i]):
            senkou_a[i + n2] = (tenkan[i] + kijun[i]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_b = np.full_like(high, np.nan)
    for i in range(len(high)):
        if i + n2 < len(high) and i >= n3-1:
            senkou_b[i + n2] = (np.max(high[i-n3+1:i+1]) + np.min(low[i-n3+1:i+1])) / 2
    
    # Chikou Span (Lagging Span): close shifted 26 periods behind
    chikou = np.full_like(high, np.nan)
    for i in range(n2, len(close)):
        chikou[i - n2] = close[i]
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
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

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Ichimoku and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Ichimoku Cloud
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d, chikou_1d = calculate_ichimoku(high_1d, low_1d, close_1d)
    
    # Calculate ADX for regime filtering
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align all daily data to daily timeframe (no alignment needed as we're already on 1d)
    tenkan = tenkan_1d
    kijun = kijun_1d
    senkou_a = senkou_a_1d
    senkou_b = senkou_b_1d
    adx = adx_1d
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(52, 26, 20) + 26  # Ensure we have enough data for Ichimoku (52) + shift (26) + ADX + volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(senkou_a[i]) or np.isnan(senkou_b[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine cloud boundaries (Senkou Span A and B)
        upper_cloud = np.maximum(senkou_a[i], senkou_b[i])
        lower_cloud = np.minimum(senkou_a[i], senkou_b[i])
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Regime filters
        trending = adx[i] > 25
        ranging = adx[i] <= 25
        
        if position == 0:
            # Long entry: price breaks above cloud in trending market OR bounces off cloud support in ranging market with volume
            if trending and close[i] > upper_cloud and vol_confirm:
                signals[i] = 0.25
                position = 1
            elif ranging and close[i] > lower_cloud and close[i] < upper_cloud and vol_confirm:
                # Additional check for bounce: price above Kijun (support) and Tenkan > Kijun (bullish momentum)
                if close[i] > kijun[i] and tenkan[i] > kijun[i]:
                    signals[i] = 0.25
                    position = 1
            # Short entry: price breaks below cloud in trending market OR bounces off cloud resistance in ranging market with volume
            elif trending and close[i] < lower_cloud and vol_confirm:
                signals[i] = -0.25
                position = -1
            elif ranging and close[i] < upper_cloud and close[i] > lower_cloud and vol_confirm:
                # Additional check for bounce: price below Kijun (resistance) and Tenkan < Kijun (bearish momentum)
                if close[i] < kijun[i] and tenkan[i] < kijun[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price breaks below cloud OR ADX drops below 20 (trend weakening) OR opposite Chikou crossover
            if close[i] < lower_cloud or adx[i] < 20 or (i >= 26 and not np.isnan(chikou_1d[i]) and close[i] < chikou_1d[i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above cloud OR ADX drops below 20 (trend weakening) OR opposite Chikou crossover
            if close[i] > upper_cloud or adx[i] < 20 or (i >= 26 and not np.isnan(chikou_1d[i]) and close[i] > chikou_1d[i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Ichimoku_Cloud_Breakout_ADX_Volume"
timeframe = "1d"
leverage = 1.0