#!/usr/bin/env python3
"""
12h Williams Alligator + 1d Volume Spike + ADX Trend Filter
Long: Alligator aligned bullish (jaw < teeth < lips), volume > 2x 20-period average, ADX > 25
Short: Alligator aligned bearish (jaw > teeth > lips), volume > 2x 20-period average, ADX > 25
Exit: Opposite Alligator alignment or volume drops below average
Williams Alligator uses SMAs of median price to identify trend alignment and avoid whipsaws.
Designed to work in both bull and bear markets by requiring strong trend (ADX>25) and volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Williams Alligator: SMAs of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-period
    
    # Get 1d data for volume average and ADX
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA(20) for volume filter
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Calculate 1d ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period] = np.nansum(tr[1:period+1]) if period < len(high) else 0
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values[i] / atr[i]) * 100
                minus_di[i] = (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros_like(high)
        adx[2*period-1] = np.nanmean(dx[period:2*period]) if 2*period <= len(high) else 0
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 13, 14)  # need Alligator components and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_sma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        vol = volume[i]
        vol_sma_val = vol_sma_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        
        # Alligator alignment conditions
        bullish_align = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_align = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        if position == 0:
            # Long: Bullish alignment + volume spike + strong trend
            if bullish_align and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + volume spike + strong trend
            elif bearish_align and vol > 2.0 * vol_sma_val and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment or volume drops below average or weak trend
            if bearish_align or vol < vol_sma_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment or volume drops below average or weak trend
            if bullish_align or vol < vol_sma_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_ADXFilter"
timeframe = "12h"
leverage = 1.0