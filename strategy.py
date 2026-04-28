#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme levels with volume confirmation and ADX trend filter.
# Enter long when 1d Williams %R < -80 (oversold) with volume spike and ADX > 25 (strong trend).
# Enter short when 1d Williams %R > -20 (overbought) with volume spike and ADX > 25.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 20-50 trades/year.
# Williams %R provides mean-reversion signals from higher timeframe, volume confirms momentum,
# ADX filter ensures we only trade in trending markets to avoid false signals in ranges.

name = "4h_WilliamsR_1d_Extreme_Volume_ADXFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and ADX (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    williams_r = np.full(n_1d, np.nan)
    
    for i in range(14, n_1d):
        highest_high = np.max(high_1d[i-14:i+1])
        lowest_low = np.min(low_1d[i-14:i+1])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_1d[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50.0
    
    # Calculate 1d ADX (14)
    def calculate_adx(high, low, close, length=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = high[i] - high[i-1] if high[i] - high[i-1] > high[i-1] - low[i-1] and high[i] - high[i-1] > 0 else 0
            minus_dm[i] = high[i-1] - low[i-1] if high[i-1] - low[i-1] > high[i] - high[i-1] and high[i-1] - low[i-1] > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[length-1] = np.mean(tr[0:length])
        plus_dm_smoothed = np.mean(plus_dm[0:length])
        minus_dm_smoothed = np.mean(minus_dm[0:length])
        
        for i in range(length, len(high)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
            plus_dm_smoothed = (plus_dm_smoothed * (length-1) + plus_dm[i]) / length
            minus_dm_smoothed = (minus_dm_smoothed * (length-1) + minus_dm[i]) / length
            
            plus_di[i] = 100 * plus_dm_smoothed / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_smoothed / atr[i] if atr[i] != 0 else 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        for i in range(length, len(high)):
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        
        adx[length*2-2] = np.mean(dx[length-1:length*2-1])
        for i in range(length*2-1, len(high)):
            adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
        
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Forward fill indicators
    williams_r = pd.Series(williams_r).ffill().values
    adx = pd.Series(adx).ffill().values
    
    # Align 1d indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h volume spike: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions with volume confirmation and ADX trend filter
        long_signal = williams_r_aligned[i] < -80 and volume_spike[i] and adx_aligned[i] > 25
        short_signal = williams_r_aligned[i] > -20 and volume_spike[i] and adx_aligned[i] > 25
        
        # Exit conditions: opposite extreme or ADX weakening
        long_exit = williams_r_aligned[i] > -50 or adx_aligned[i] < 20
        short_exit = williams_r_aligned[i] < -50 or adx_aligned[i] < 20
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals