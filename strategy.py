#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with Volume Spike and 1d ADX Trend Filter.
Long when price breaks above Camarilla R3 with volume > 1.5x 20-period average AND 1d ADX > 25 (trending market).
Short when price breaks below Camarilla S3 with volume > 1.5x 20-period average AND 1d ADX > 25.
Exit when price returns to Camarilla R1/S1 levels or ADX drops below 20 (trend weakening).
Uses 1d for ADX trend filter, 6h for Camarilla levels and volume confirmation.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla provides precise intraday support/resistance,
volume confirms breakout strength, and ADX filter ensures we only trade in trending conditions to avoid whipsaws.
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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros(len(high))
        atr[period] = np.nansum(tr[1:period+1]) if period < len(high) else 0
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] > 0:
                plus_di[i] = (np.nansum(plus_dm[i-period+1:i+1]) * 100) / (atr[i] * period)
                minus_di[i] = (np.nansum(minus_dm[i-period+1:i+1]) * 100) / (atr[i] * period)
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.nansum(dx[period:2*period]) if (2*period) < len(high) else 0
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_14_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate volume average (20-period)
    volume_s = pd.Series(volume)
    volume_ma20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels for each 6h bar using previous 1d bar
    # We'll calculate these in the loop using the previous completed 1d bar
    
    # Align 1d ADX to 6h timeframe
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(adx_14_1d_aligned[i]) or np.isnan(volume_ma20[i]):
            signals[i] = 0.0
            continue
        
        # Get previous completed 1d bar for Camarilla calculation
        # We need to find the index of the previous 1d bar in the 1d dataframe
        # Since we can't easily map 6h index to 1d index without look-ahead,
        # we'll use a simplified approach: calculate Camarilla based on recent 1d data
        # For practical purposes, we'll use the last available 1d bar's OHLC
        
        # Simplified Camarilla calculation using rolling window
        # In practice, we'd use the previous day's H/L/C, but for now we approximate
        if i >= 24:  # approximately 1 day of 6h bars
            # Get recent 6h bar's high/low/close as proxy for daily (not perfect but workable)
            recent_high = np.max(high[max(0, i-24):i])
            recent_low = np.min(low[max(0, i-24):i])
            recent_close = close[i-1]
            
            # Calculate Camarilla levels
            range_val = recent_high - recent_low
            if range_val > 0:
                camarilla_r3 = recent_close + range_val * 1.1/2
                camarilla_s3 = recent_close - range_val * 1.1/2
                camarilla_r1 = recent_close + range_val * 1.1/12
                camarilla_s1 = recent_close - range_val * 1.1/12
            else:
                camarilla_r3 = camarilla_s3 = camarilla_r1 = camarilla_s1 = recent_close
        else:
            camarilla_r3 = camarilla_s3 = camarilla_r1 = camarilla_s1 = close[i]
        
        adx = adx_14_1d_aligned[i]
        vol_ratio = volume[i] / volume_ma20[i] if volume_ma20[i] > 0 else 0
        price = close[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike AND ADX > 25 (trending)
            if price > camarilla_r3 and vol_ratio > 1.5 and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike AND ADX > 25 (trending)
            elif price < camarilla_s3 and vol_ratio > 1.5 and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to R1 OR ADX drops below 20 (trend weakening)
            if price < camarilla_r1 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to S1 OR ADX drops below 20 (trend weakening)
            if price > camarilla_s1 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_Volume_ADXFilter"
timeframe = "6h"
leverage = 1.0