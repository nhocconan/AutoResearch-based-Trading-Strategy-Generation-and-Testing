#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d ATR volatility filter + volume confirmation
# Donchian breakouts capture momentum; 1d ATR filter ensures breakouts occur during sufficient volatility
# Volume confirmation validates breakout authenticity
# Works in bull/bear: ATR filter adapts to changing market conditions, Donchian breakouts work in both directions
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_donchian_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_1d = np.full(len(tr), np.nan)
    for i in range(len(tr)):
        if i < 1:
            atr_1d[i] = np.nan
        elif i < 14:
            # Use simple average for first 14 periods
            valid_tr = tr[1:i+1]  # Exclude first NaN
            if len(valid_tr) > 0:
                atr_1d[i] = np.mean(valid_tr)
            else:
                atr_1d[i] = np.nan
        else:
            # Wilder's smoothing: ATR[i] = (ATR[i-1] * 13 + TR[i]) / 14
            if not np.isnan(atr_1d[i-1]) and not np.isnan(tr[i]):
                atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
            else:
                atr_1d[i] = np.nan
    
    # Align 1d ATR to 12h timeframe (wait for daily close)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # ATR filter: current ATR > 0.5 * 20-period average ATR (ensure sufficient volatility)
        if i >= 20:
            atr_ma = np.nanmean(atr_1d_aligned[max(0, i-20):i])
            atr_filter = not np.isnan(atr_ma) and atr_1d_aligned[i] > 0.5 * atr_ma
        else:
            atr_filter = False
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low (trend reversal)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high (trend reversal)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with ATR filter, volume confirmation and Donchian breakout
            if atr_filter and volume_confirmed:
                # Long entry: price > Donchian high (bullish breakout)
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low (bearish breakout)
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals