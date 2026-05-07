# 4h_Structure_Breakout_Volume
# Hypothesis: Price breaking above/below 4h structure (recent swing high/low) with volume confirmation and ADX trend filter.
# Uses 1d ADX for trend strength and 4h volume spike (>2x average) to filter entries. Designed for 20-40 trades/year
# to minimize fee drag while capturing trend continuations in both bull and bear markets.
# Timeframe: 4h, HTF: 1d for trend filter.

#!/usr/bin/env python3
name = "4h_Structure_Breakout_Volume"
timeframe = "4h"
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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h swing points: recent swing high/low (10-period lookback)
    def find_swing_points(high, low, lookback=10):
        swing_high = np.full_like(high, np.nan)
        swing_low = np.full_like(low, np.nan)
        for i in range(lookback, len(high)):
            if high[i] == np.max(high[i-lookback:i+1]):
                swing_high[i] = high[i]
            if low[i] == np.min(low[i-lookback:i+1]):
                swing_low[i] = low[i]
        return swing_high, swing_low
    
    swing_high_4h, swing_low_4h = find_swing_points(high, low, 10)
    
    # 1d ADX for trend filter (ADX > 25 indicates strong trend)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_dm_smooth = np.zeros_like(plus_dm)
        minus_dm_smooth = np.zeros_like(minus_dm)
        
        atr[period] = np.nansum(tr[1:period+1])
        plus_dm_smooth[period] = np.nansum(plus_dm[1:period+1])
        minus_dm_smooth[period] = np.nansum(minus_dm[1:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1] / period) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1] / period) + minus_dm[i]
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        
        # Smooth DX to get ADX
        adx = np.zeros_like(dx)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
        for i in range(2*period+1, len(dx)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 4h volume spike: > 2.0x 20-period average
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike_4h = volume > 2.0 * vol_ma_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 34)  # Wait for swing points, ADX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(swing_high_4h[i]) or np.isnan(swing_low_4h[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above recent swing high with volume spike and strong trend
            if (close[i] > swing_high_4h[i] and vol_spike_4h[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Break below recent swing low with volume spike and strong trend
            elif (close[i] < swing_low_4h[i] and vol_spike_4h[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below recent swing low or trend weakening (ADX < 20)
            if close[i] < swing_low_4h[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above recent swing high or trend weakening (ADX < 20)
            if close[i] > swing_high_4h[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: Uses 4h swing points for structure breaks, 1d ADX for trend filter, and volume spike for confirmation.
# Position size 0.25 limits risk. Target 20-40 trades/year to minimize fee drag.
# Exit on retrace to swing point or trend weakening (ADX < 20).