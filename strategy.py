#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Williams Alligator with 1-day volume confirmation and ADX trend filter.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long when Lips > Teeth > Jaw and ADX > 25 (trending up)
- Short when Lips < Teeth < Jaw and ADX > 25 (trending down)
- Volume confirmation: current volume > 1.5x 20-period volume moving average
- Fixed position size 0.25 to manage drawdown
- Designed for 4h timeframe with strict entry conditions to limit trades to 75-200 total over 4 years
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    result = np.full_like(arr, np.nan, dtype=np.float64)
    sma = np.mean(arr[:period])
    result[period-1] = sma
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume moving average (20-period)
    volume_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_20_1d)
    
    # Williams Alligator components (using SMMA)
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the Alligator lines as per Williams
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set initial values to NaN for shifted periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period * 2:
            return np.full_like(high, np.nan)
        
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0  # First value has no previous close
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        plus_dm[0] = 0
        minus_dm[0] = 0
        
        # Smoothed TR, PlusDM, MinusDM
        tr_period = np.zeros_like(tr)
        plus_dm_period = np.zeros_like(plus_dm)
        minus_dm_period = np.zeros_like(minus_dm)
        
        tr_period[period-1] = np.sum(tr[:period])
        plus_dm_period[period-1] = np.sum(plus_dm[:period])
        minus_dm_period[period-1] = np.sum(minus_dm[:period])
        
        for i in range(period, len(tr)):
            tr_period[i] = tr_period[i-1] - (tr_period[i-1] / period) + tr[i]
            plus_dm_period[i] = plus_dm_period[i-1] - (plus_dm_period[i-1] / period) + plus_dm[i]
            minus_dm_period[i] = minus_dm_period[i-1] - (minus_dm_period[i-1] / period) + minus_dm[i]
        
        # Directional Indicators
        plus_di = 100 * plus_dm_period / tr_period
        minus_di = 100 * minus_dm_period / tr_period
        
        # DX and ADX
        dx = np.zeros_like(close)
        dx[period-1:] = 100 * np.abs(plus_di[period-1:] - minus_di[period-1:]) / (plus_di[period-1:] + minus_di[period-1:])
        adx = np.zeros_like(close)
        adx[2*period-2:] = np.nan
        
        # Smoothed DX for ADX
        dx_period = np.zeros_like(dx)
        dx_period[2*period-2] = np.sum(dx[period-1:2*period-1])
        for i in range(2*period-1, len(dx)):
            dx_period[i] = dx_period[i-1] - (dx_period[i-1] / period) + dx[i]
        
        adx[2*period-2:] = 100 * dx_period[2*period-2:] / dx_period[2*period-2:]
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 30  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(jaw_shifted[i]) or
            np.isnan(adx[i]) or np.isnan(volume_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_20_1d_aligned[i]
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        adx_val = adx[i]
        
        if position == 0:
            # Look for Alligator alignment with ADX trend filter and volume confirmation
            # Long: Lips > Teeth > Jaw (bullish alignment) and ADX > 25 (strong trend)
            if lips_val > teeth_val > jaw_val and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) and ADX > 25 (strong trend)
            elif lips_val < teeth_val < jaw_val and adx_val > 25 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when Alligator loses bullish alignment or ADX weakens
            if not (lips_val > teeth_val > jaw_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when Alligator loses bearish alignment or ADX weakens
            if not (lips_val < teeth_val < jaw_val) or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ADX_Volume"
timeframe = "4h"
leverage = 1.0