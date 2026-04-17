#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ADX Regime Filter.
Long when Jaw < Teeth < Lips (bullish alignment) AND 1d ADX > 25 (trending market).
Short when Jaw > Teeth > Lips (bearish alignment) AND 1d ADX > 25 (trending market).
Exit when Alligator lines converge (|Lips - Jaw| < 0.1 * ATR) or ADX < 20 (range market).
Uses 1d for ADX regime filter, 12h for Alligator calculation.
Target: 50-150 total trades over 4 years (12-37/year). Alligator identifies trend direction and strength,
while 1d ADX ensures we only trade in strong trending conditions to avoid whipsaws in ranging markets.
Works in both bull (catch trends) and bear (avoid false signals in chop) markets.
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX for regime filter
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (np.zeros_like(plus_dm))
        minus_di = 100 * (np.zeros_like(minus_dm))
        
        # Smooth +DM and -DM
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        plus_dm_smooth[period] = np.nanmean(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nanmean(minus_dm[1:period+1])
        
        for i in range(period+1, len(plus_dm)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        plus_di = 100 * plus_dm_smooth / (atr + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr + 1e-10)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.zeros_like(dx)
        adx[2*period-1] = np.nanmean(dx[period:2*period])
        for i in range(2*period, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (Blue): 13-period SMMA smoothed 8 periods ahead
    # Teeth (Red): 8-period SMMA smoothed 5 periods ahead  
    # Lips (Green): 5-period SMMA smoothed 3 periods ahead
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan)
        if len(source) >= period:
            result[period-1] = np.mean(source[:period])
            for i in range(period, len(source)):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Calculate SMMA for different periods
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply smoothing offsets (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    for i in range(8, len(jaw)):
        jaw[i] = jaw_raw[i-8] if i-8 >= 0 and not np.isnan(jaw_raw[i-8]) else np.nan
    for i in range(5, len(teeth)):
        teeth[i] = teeth_raw[i-5] if i-5 >= 0 and not np.isnan(teeth_raw[i-5]) else np.nan
    for i in range(3, len(lips)):
        lips[i] = lips_raw[i-3] if i-3 >= 0 and not np.isnan(lips_raw[i-3]) else np.nan
    
    # Calculate 12h ATR for convergence check
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = np.zeros_like(tr)
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_12h = calculate_atr(high, low, close, 14)
    
    # Align 1d ADX to 12h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
        
        j = jaw[i]
        t = teeth[i]
        l = lips[i]
        adx = adx_1d_aligned[i]
        atr = atr_12h[i]
        
        if position == 0:
            # Long: Jaw < Teeth < Lips (bullish alignment) AND ADX > 25 (strong trend)
            if j < t < l and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: Jaw > Teeth > Lips (bearish alignment) AND ADX > 25 (strong trend)
            elif j > t > l and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator convergence OR ADX < 20 (losing trend strength)
            convergence = abs(l - j) < 0.1 * atr if not np.isnan(atr) else False
            if convergence or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator convergence OR ADX < 20 (losing trend strength)
            convergence = abs(l - j) < 0.1 * atr if not np.isnan(atr) else False
            if convergence or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dADX_Regime"
timeframe = "12h"
leverage = 1.0