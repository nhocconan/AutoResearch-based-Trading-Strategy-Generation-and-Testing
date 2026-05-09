#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (3 SMAs) with 12h ADX trend filter and volume confirmation.
# Alligator identifies trend direction via Jaw/Teeth/Lips alignment; ADX filters weak trends.
# Works in bull/bear by only trading when trend is strong (ADX > 25). Volume confirms breakout strength.
# Target: 50-150 trades over 4 years (12-37/year) to avoid fee drag.

name = "6h_Alligator_ADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
    # SMMA = smoothed moving average (similar to Wilder's smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        res = np.full_like(arr, np.nan, dtype=float)
        sma = np.mean(arr[:period])
        res[period-1] = sma
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    close_12h = df_12h['close'].values
    jaw = smma(close_12h, 13)  # Blue line
    teeth = smma(close_12h, 8)  # Red line
    lips = smma(close_12h, 5)   # Green line
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Invalidate shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(close, np.nan, dtype=float)
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        # Initial values
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_sum = np.sum(plus_dm[1:period+1])
        minus_dm_sum = np.sum(minus_dm[1:period+1])
        plus_di[period] = 100 * plus_dm_sum / atr[period] if atr[period] != 0 else 0
        minus_di[period] = 100 * minus_dm_sum / atr[period] if atr[period] != 0 else 0
        # Wilder smoothing
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_di[i] = 100 * ((plus_di[i-1] * (period-1) + plus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * ((minus_di[i-1] * (period-1) + minus_dm[i]) / period) / atr[i] if atr[i] != 0 else 0
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        dx[np.isnan(plus_di) | np.isnan(minus_di) | (plus_di + minus_di) == 0] = np.nan
        adx = np.zeros_like(high)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align Alligator lines and ADX to 6h
    jaw_6h = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_6h = align_htf_to_ltf(prices, df_12h, teeth)
    lips_6h = align_htf_to_ltf(prices, df_12h, lips)
    adx_6h = align_htf_to_ltf(prices, df_12h, adx)
    
    # Volume filter: above 1.5x 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA and ADX
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_6h[i]) or np.isnan(teeth_6h[i]) or np.isnan(lips_6h[i]) or 
            np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        trend_ok = adx_6h[i] > 25  # Strong trend
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend; Lips < Teeth < Jaw = downtrend
        lips_above_teeth = lips_6h[i] > teeth_6h[i]
        teeth_above_jaw = teeth_6h[i] > jaw_6h[i]
        lips_below_teeth = lips_6h[i] < teeth_6h[i]
        teeth_below_jaw = teeth_6h[i] < jaw_6h[i]
        
        if position == 0:
            # Long: Alligator aligned up + strong trend + volume
            if lips_above_teeth and teeth_above_jaw and trend_ok and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + strong trend + volume
            elif lips_below_teeth and teeth_below_jaw and trend_ok and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks (Lips cross below Teeth) or trend weakens
            if not (lips_above_teeth and teeth_above_jaw) or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks (Lips cross above Teeth) or trend weakens
            if not (lips_below_teeth and teeth_below_jaw) or adx_6h[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals