#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_v1
Hypothesis: Use 1d Camarilla pivot levels (R1/S1) as key reversal zones on 12h timeframe. 
Enter long when price touches S1 and shows bullish reversal (close > open), short when price touches R1 with bearish reversal (close < open). 
Require volume > 1.3x 20-period average for confirmation. 
Apply 1d ADX(14) > 25 filter to ensure trending environment (avoid chop). 
Target: 15-30 trades/year by focusing on high-probability pivot reversals in trending markets.
Works in bull via buying S1 bounces, in bear via selling R1 rejections.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # Using previous day's values to avoid look-ahead
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # First day will have invalid values (from roll), handled by isnan check later
    
    rng = prev_high_1d - prev_low_1d
    R1 = prev_close_1d + 1.1 * rng / 12.0
    S1 = prev_close_1d - 1.1 * rng / 12.0
    
    # Calculate 1d ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        # Smooth TR and DM
        tr_period = np.full_like(tr, np.nan)
        dm_plus_period = np.full_like(dm_plus, np.nan)
        dm_minus_period = np.full_like(dm_minus, np.nan)
        if len(tr) >= period:
            tr_period[period-1] = np.nansum(tr[:period])
            dm_plus_period[period-1] = np.nansum(dm_plus[:period])
            dm_minus_period[period-1] = np.nansum(dm_minus[:period])
            for i in range(period, len(tr)):
                tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
                dm_plus_period[i] = dm_plus_period[i-1] - (dm_plus_period[i-1] / period) + dm_plus[i]
                dm_minus_period[i] = dm_minus_period[i-1] - (dm_minus_period[i-1] / period) + dm_minus[i]
        # DI and DX
        plus_di = 100 * dm_plus_period / tr_period
        minus_di = 100 * dm_minus_period / tr_period
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        # ADX = smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(dx) >= period:
            adx[2*period-2] = np.nanmean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(dx)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 12h timeframe
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1) + 1  # Need volume MA and at least 2 days for pivots
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in trending markets (ADX > 25)
        strong_trend = adx_12h[i] > 25
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Price near pivot levels (within 0.1% tolerance)
        tol = 0.001  # 0.1%
        near_S1 = abs(close[i] - S1_12h[i]) / S1_12h[i] < tol
        near_R1 = abs(close[i] - R1_12h[i]) / R1_12h[i] < tol
        
        # Reversal signals
        bullish_reversal = close[i] > open_price[i]  # bullish candle
        bearish_reversal = close[i] < open_price[i]  # bearish candle
        
        if position == 0 and strong_trend and vol_confirm:
            # Long: price near S1 with bullish reversal
            if near_S1 and bullish_reversal:
                signals[i] = 0.25
                position = 1
            # Short: price near R1 with bearish reversal
            elif near_R1 and bearish_reversal:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses R1 or shows bearish reversal at resistance
            if close[i] > R1_12h[i] or (near_R1 and bearish_reversal):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses S1 or shows bullish reversal at support
            if close[i] < S1_12h[i] or (near_S1 and bullish_reversal):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_v1"
timeframe = "12h"
leverage = 1.0