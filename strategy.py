#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Parabolic SAR with 1-day ADX filter and volume confirmation.
# Parabolic SAR provides trend-following signals with built-in stop/reverse mechanism.
# 1-day ADX > 25 filters for trending markets to avoid whipsaws in ranging conditions.
# Volume confirmation (>1.5x 20-period average) reduces false signals.
# Designed to work in both bull and bear markets by using 1d trend filter to avoid counter-trend trades.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for PSAR and ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Parabolic SAR on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Initialize SAR
    psar = np.zeros_like(high_1d)
    bull = True  # Start assuming uptrend
    af = 0.02    # Acceleration factor
    max_af = 0.2
    ep = high_1d[0] if bull else low_1d[0]  # Extreme point
    psar[0] = low_1d[0] if bull else high_1d[0]
    
    for i in range(1, len(high_1d)):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR stays within prior period's range
            psar[i] = min(psar[i], min(high_1d[i-1], low_1d[i-1]))
            # Reverse conditions
            if low_1d[i] < psar[i]:
                bull = False
                psar[i] = ep
                ep = low_1d[i]
                af = 0.02
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            # Ensure SAR stays within prior period's range
            psar[i] = max(psar[i], max(high_1d[i-1], low_1d[i-1]))
            # Reverse conditions
            if high_1d[i] > psar[i]:
                bull = True
                psar[i] = ep
                ep = high_1d[i]
                af = 0.02
        
        # Update extreme point and acceleration factor
        if bull:
            if high_1d[i] > ep:
                ep = high_1d[i]
                af = min(af + 0.02, max_af)
        else:
            if low_1d[i] < ep:
                ep = low_1d[i]
                af = min(af + 0.02, max_af)
    
    # Calculate ADX on 1d data
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align indicators to 12h timeframe
    psar_aligned = align_htf_to_ltf(prices, df_1d, psar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # Need enough data for PSAR and ADX
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(psar_aligned[i]) or 
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        if position == 0:
            # Look for PSAR signals
            # Only trade in trending markets
            
            # Long: price above PSAR AND trending market
            if (close[i] > psar_aligned[i] and 
                trending and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price below PSAR AND trending market
            elif (close[i] < psar_aligned[i] and 
                  trending and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below PSAR (SAR flip) or trend weakens
            if (close[i] < psar_aligned[i] or 
                adx_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above PSAR (SAR flip) or trend weakens
            if (close[i] > psar_aligned[i] or 
                adx_aligned[i] < 20):  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dPSAR_1dADX_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0