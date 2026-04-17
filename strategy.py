#!/usr/bin/env python3
"""
12h Williams Alligator + 1d Volume Spike + ADX Trend Filter
Long: Green line > Red line > Blue line, price > Green line, volume > 1.5x 12h volume SMA(20), ADX > 20
Short: Red line > Blue line > Green line, price < Red line, volume > 1.5x 12h volume SMA(20), ADX > 20
Exit: Opposite Alligator alignment or ADX < 15
Williams Alligator uses smoothed medians to identify trend direction and avoid whipsaws.
Designed for low-frequency trading on 12h timeframe to minimize fee drag.
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
    
    # Get 1d data for volume spike filter (using 12h volume but checking against 1d average for regime)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    # Get 1d data for ADX trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d
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
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period-1] = np.nansum(tr[:period])
        plus_dm_smooth[period-1] = np.nansum(plus_dm[:period])
        minus_dm_smooth[period-1] = np.nansum(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = np.zeros_like(dx)
        adx[2*period-2] = np.nansum(dx[:2*period-1]) / period if np.nansum(dx[:2*period-1]) > 0 else 0
        
        for i in range(2*period-1, len(high)):
            adx[i] = adx[i-1] - (adx[i-1] / period) + dx[i]
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Williams Alligator on 12h prices
    # Jaw (Blue Line): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red Line): 8-period SMMA, shifted 5 bars forward
    # Lips (Green Line): 5-period SMMA, shifted 3 bars forward
    def smma(series, period):
        sma = np.zeros_like(series)
        sma[period-1] = np.mean(series[:period])
        for i in range(period, len(series)):
            sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Shift the lines (Alligator specific)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set initial values to NaN for shifted periods
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 12h volume SMA for volume spike filter
    vol_sma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(20, 30)  # need sufficient data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(vol_sma_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(vol_sma_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_12h_val = vol_sma_12h[i]
        vol_sma_1d_val = vol_sma_1d_aligned[i]
        adx_val = adx_1d_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Alligator alignments
        bullish_alignment = lips_val > teeth_val > jaw_val  # Green > Red > Blue
        bearish_alignment = jaw_val > teeth_val > lips_val  # Blue > Red > Green
        
        if position == 0:
            # Long: Bullish alignment + price above lips (green line) + volume spike + ADX > 20
            if (bullish_alignment and price > lips_val and 
                vol > 1.5 * vol_sma_12h_val and vol > vol_sma_1d_val and adx_val > 20):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + price below jaws (blue line) + volume spike + ADX > 20
            elif (bearish_alignment and price < jaw_val and 
                  vol > 1.5 * vol_sma_12h_val and vol > vol_sma_1d_val and adx_val > 20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment OR ADX < 15 (trend weakening) OR price below teeth
            if bearish_alignment or adx_val < 15 or price < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment OR ADX < 15 (trend weakening) OR price above teeth
            if bullish_alignment or adx_val < 15 or price > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolume_ADXFilter"
timeframe = "12h"
leverage = 1.0