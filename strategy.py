# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h ADX + Williams Alligator with 1d trend filter
- ADX > 25 indicates trending market (avoid chop)
- Williams Alligator (Jaw/Teeth/Lips) provides directional signal
- 1d EMA(50) as higher timeframe trend filter to avoid counter-trend trades
- Works in bull (follow uptrend) and bear (follow downtrend) via ADX + Alligator alignment
- Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
"""
name = "6h_ADX_Alligator_1dTrendFilter"
timeframe = "6h"
leverage = 1.0

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
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of median price
    # Jaw: SMMA(13, 8), Teeth: SMMA(8, 5), Lips: SMMA(5, 3)
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        """Average Directional Index"""
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr = np.zeros_like(tr)
        atr[:period] = np.nan
        atr[period] = np.mean(tr[:period+1])
        
        dm_plus_smooth = np.zeros_like(dm_plus)
        dm_minus_smooth = np.zeros_like(dm_minus)
        dm_plus_smooth[:period] = np.nan
        dm_minus_smooth[:period] = np.nan
        dm_plus_smooth[period] = np.mean(dm_plus[:period+1])
        dm_minus_smooth[period] = np.mean(dm_minus[:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (period-1) + dm_plus[i]) / period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (period-1) + dm_minus[i]) / period
        
        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / atr
        minus_di = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx[:] = np.nan
        dx[2*period:] = 100 * np.abs(plus_di[2*period:] - minus_di[2*period:]) / (plus_di[2*period:] + minus_di[2*period:])
        
        adx = np.zeros_like(close)
        adx[:] = np.nan
        adx[3*period:] = np.nan
        # First ADX value is average of first 'period' DX values
        adx[2*period + period - 1] = np.mean(dx[2*period:2*period+period])
        for i in range(2*period + period, len(close)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from sufficient lookback
    start_idx = max(50, 3*14)  # Ensure ADX and Alligator are valid
    
    for i in range(start_idx, n):
        # Skip if any required values are NaN
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ADX > 25 indicates trending market
        trending = adx[i] > 25
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_up = lips[i] > teeth[i] > jaw[i]
        alligator_down = lips[i] < teeth[i] < jaw[i]
        
        # 1d trend filter: price relative to EMA50
        price_above_1d = close[i] > ema50_1d_aligned[i]
        price_below_1d = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # LONG: ADX trending + Alligator up + price above 1d EMA
            if trending and alligator_up and price_above_1d:
                signals[i] = 0.25
                position = 1
            # SHORT: ADX trending + Alligator down + price below 1d EMA
            elif trending and alligator_down and price_below_1d:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Alligator reverses or ADX weakens
            if not alligator_up or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator reverses or ADX weakens
            if not alligator_down or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals