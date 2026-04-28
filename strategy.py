#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes with volume confirmation and ADX trend filter.
# Enter long when 1d Williams %R < -80 (oversold) with volume spike and ADX > 25 (trending).
# Enter short when 1d Williams %R > -20 (overbought) with volume spike and ADX > 25.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 20-50 trades/year.
# Williams %R identifies reversal points in higher timeframe, volume confirms momentum,
# ADX filter ensures trades only in trending markets to avoid chop losses.
# Works in bull (buy oversold dips) and bear (sell overbought rallies) markets.

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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    n_1d = len(high_1d)
    
    # Calculate 1d Williams %R (14)
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
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            if plus_dm[i] < 0: plus_dm[i] = 0
            if minus_dm[i] < 0: minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        # Initial values
        atr[length] = np.mean(tr[1:length+1])
        plus_dm_sum = np.sum(plus_dm[1:length+1])
        minus_dm_sum = np.sum(minus_dm[1:length+1])
        
        plus_di[length] = 100 * plus_dm_sum / (atr[length] * length) if atr[length] != 0 else 0
        minus_di[length] = 100 * minus_dm_sum / (atr[length] * length) if atr[length] != 0 else 0
        dx[length] = 100 * abs(plus_di[length] - minus_di[length]) / (plus_di[length] + minus_di[length]) if (plus_di[length] + minus_di[length]) != 0 else 0
        
        # Smooth subsequent values
        for i in range(length+1, len(high)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
            plus_di[i] = 100 * (plus_dm[i] + plus_di[i-1] * (length-1)) / (atr[i] * length) if atr[i] != 0 else 0
            minus_di[i] = 100 * (minus_dm[i] + minus_di[i-1] * (length-1)) / (atr[i] * length) if atr[i] != 0 else 0
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
            if i >= 2*length:
                adx[i] = (adx[i-1] * (length-1) + dx[i]) / length
        
        # Initialize first values
        for i in range(length, 2*length):
            if i < len(adx):
                adx[i] = dx[i]
        
        return adx
    
    adx = calculate_adx(high_1d, low_1d, close_1d, 14)
    
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
        
        # Exit conditions: opposite extreme or loss of trend
        long_exit = williams_r_aligned[i] > -20 or adx_aligned[i] < 20
        short_exit = williams_r_aligned[i] < -80 or adx_aligned[i] < 20
        
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