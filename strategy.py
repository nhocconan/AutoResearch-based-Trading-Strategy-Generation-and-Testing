#!/usr/bin/env python3
"""
4h Williams Alligator + 1d Volume Spike + ADX Trend Filter
Long: Jaw < Teeth < Lips (bullish alignment) + volume > 2x 1d avg volume + ADX > 25
Short: Jaw > Teeth > Lips (bearish alignment) + volume > 2x 1d avg volume + ADX > 25
Exit: Opposite Alligator alignment or ADX < 20
Williams Alligator identifies trend alignment; volume confirms conviction; ADX filters ranging markets.
Designed to work in both bull and bear markets by requiring strong trend (ADX>25).
Target: 50-150 total trades over 4 years (12-37/year)
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
    
    # Get 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d average volume (simple mean of last 20 days)
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Williams Alligator: SMAs with future shift (using only past data)
    # Jaw: 13-period SMMA smoothed 8 bars ahead -> we use 13 SMA then shift 8
    # Teeth: 8-period SMMA smoothed 5 bars ahead -> 8 SMA then shift 5
    # Lips: 5-period SMMA smoothed 3 bars ahead -> 5 SMA then shift 3
    # Since we can't use future data, we calculate SMMA and use only completed values
    def smma(series, period):
        """Smoothed Moving Average"""
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma = np.full_like(series, np.nan, dtype=float)
        smma[period-1] = sma[period-1]
        for i in range(period, len(series)):
            if not np.isnan(sma[i]):
                smma[i] = (smma[i-1] * (period-1) + sma[i]) / period
        return smma
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply the smoothing offset by using only values that are "complete"
    # Jaw values are valid starting at index 13+8-1 = 20
    # Teeth values are valid starting at index 8+5-1 = 12
    # Lips values are valid starting at index 5+3-1 = 7
    jaw = np.full_like(jaw_raw, np.nan)
    teeth = np.full_like(teeth_raw, np.nan)
    lips = np.full_like(lips_raw, np.nan)
    
    # Shift the smoothed values to represent the Alligator lines properly
    # We use the smoothed values but consider them valid only after their smoothing period
    jaw[20:] = jaw_raw[20:]  # Jaw valid from index 20
    teeth[12:] = teeth_raw[12:]  # Teeth valid from index 12
    lips[7:] = lips_raw[7:]  # Lips valid from index 7
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = np.zeros(len(high))
        plus_dm_smooth = np.zeros(len(high))
        minus_dm_smooth = np.zeros(len(high))
        
        # Initial values
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
            
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        dx = np.zeros(len(high))
        for i in range(len(high)):
            if atr[i] != 0:
                dx[i] = (abs(plus_dm_smooth[i] - minus_dm_smooth[i]) / (plus_dm_smooth[i] + minus_dm_smooth[i])) * 100
            else:
                dx[i] = 0
        
        # ADX is smoothed DX
        adx = np.full_like(dx, np.nan)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(21, 20)  # Need Alligator and ADX values
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(adx[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol_avg_1d_val = vol_avg_1d_aligned[i]
        vol = volume[i]
        
        # Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long: Bullish alignment + volume spike + strong trend
            if bullish_alignment and vol > 2.0 * vol_avg_1d_val and adx[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike + strong trend
            elif bearish_alignment and vol > 2.0 * vol_avg_1d_val and adx[i] > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or weak trend
            if bearish_alignment or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or weak trend
            if bullish_alignment or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dVolumeSpike_ADXFilter"
timeframe = "4h"
leverage = 1.0