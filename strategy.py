#!/usr/bin/env python3
"""
12h_Bollinger_Bandwidth_Trend_Follow
Hypothesis: In trending markets (Bollinger Bandwidth > 30th percentile), trade breakouts of 20-period Bollinger Bands with volume confirmation.
In ranging markets (low bandwidth), stay flat. Uses 1-day ADX to confirm trend strength and avoid false breakouts.
Designed for 12h timeframe to target 15-30 trades/year with high-conviction entries.
Works in bull markets by capturing upside breakouts and in bear markets by capturing downside breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands: upper, lower, bandwidth"""
    sma = np.zeros_like(close)
    std = np.zeros_like(close)
    
    for i in range(len(close)):
        if i >= period - 1:
            sma[i] = np.mean(close[i-period+1:i+1])
            std[i] = np.std(close[i-period+1:i+1])
        else:
            sma[i] = np.nan
            std[i] = np.nan
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma * 100  # percentage
    return upper, lower, bandwidth

def calculate_adx(high, low, close, period=14):
    """Calculate Average Directional Index"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    
    # Smooth TR, DM+
    tr14 = np.zeros_like(tr)
    dm_plus_14 = np.zeros_like(dm_plus)
    dm_minus_14 = np.zeros_like(dm_minus)
    
    if len(tr) >= period:
        tr14[period-1] = np.sum(tr[:period])
        dm_plus_14[period-1] = np.sum(dm_plus[:period])
        dm_minus_14[period-1] = np.sum(dm_minus[:period])
        
        for i in range(period, len(tr)):
            tr14[i] = tr14[i-1] - (tr14[i-1] / period) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / period) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / period) + dm_minus[i]
    
    # Directional Indicators
    di_plus = np.zeros_like(tr)
    di_minus = np.zeros_like(tr)
    dx = np.zeros_like(tr)
    
    for i in range(period-1, len(tr)):
        if tr14[i] != 0:
            di_plus[i] = 100 * dm_plus_14[i] / tr14[i]
            di_minus[i] = 100 * dm_minus_14[i] / tr14[i]
            if di_plus[i] + di_minus[i] != 0:
                dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
    
    # ADX
    adx = np.zeros_like(tr)
    if len(tr) >= 2 * period - 1:
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(tr)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ADX for trend strength filter
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Bollinger Bands on 12h price
    close = prices['close'].values
    upper, lower, bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(adx_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(bandwidth[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: ADX > 25 indicates trending market
        trend_ok = adx_1d_aligned[i] > 25
        
        # Bandwidth filter: avoid extremely low volatility (choppy markets)
        if i >= 50:
            bw_threshold = np.percentile(bandwidth[:i+1], 30)
            bw_ok = bandwidth[i] > bw_threshold
        else:
            bw_ok = True
        
        if position == 0:
            # Long: price breaks above upper Bollinger Band with volume and trend confirmation
            if (price > upper[i] and volume_ok and trend_ok and bw_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Bollinger Band with volume and trend confirmation
            elif (price < lower[i] and volume_ok and trend_ok and bw_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle (SMA) or trend weakens
            sma_20 = np.mean(close[max(0, i-19):i+1])
            if price < sma_20 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle (SMA) or trend weakens
            sma_20 = np.mean(close[max(0, i-19):i+1])
            if price > sma_20 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Bollinger_Bandwidth_Trend_Follow"
timeframe = "12h"
leverage = 1.0