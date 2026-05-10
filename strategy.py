#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Reversal_With_Volume
# Hypothesis: In ranging markets, price tends to revert from Camarilla pivot levels (H3/L3).
# In trending markets, price breaks through H4/L4 with momentum. We use 1d ADX to detect regime:
# ADX > 25 = trend (breakout strategy), ADX <= 25 = range (mean reversion).
# Volume confirmation filters false signals. Works in both bull and bear markets by adapting to regime.

name = "4h_Camarilla_Pivot_Reversal_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX regime filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original index
        
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = np.full_like(high, np.nan, dtype=float)
        dm_plus_smooth = np.full_like(high, np.nan, dtype=float)
        dm_minus_smooth = np.full_like(high, np.nan, dtype=float)
        
        # Wilder's smoothing (EMA with alpha=1/period)
        atr[period] = np.nansum(tr[1:period+1])
        dm_plus_smooth[period] = np.nansum(dm_plus[1:period+1])
        dm_minus_smooth[period] = np.nansum(dm_minus[1:period+1])
        
        for i in range(period + 1, len(high)):
            atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
            dm_plus_smooth[i] = dm_plus_smooth[i-1] - (dm_plus_smooth[i-1] / period) + dm_plus[i]
            dm_minus_smooth[i] = dm_minus_smooth[i-1] - (dm_minus_smooth[i-1] / period) + dm_minus[i]
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / atr
        di_minus = 100 * dm_minus_smooth / atr
        
        # DX and ADX
        dx = np.full_like(high, np.nan, dtype=float)
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        adx = np.full_like(high, np.nan, dtype=float)
        adx[2*period] = np.nansum(dx[period+1:2*period+1]) / period
        for i in range(2*period + 1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    # Calculate Camarilla levels from previous day
    def calculate_camarilla(high, low, close):
        # Typical price
        tp = (high + low + close) / 3
        # Camarilla levels
        H4 = tp + 1.1 * (high - low) / 2
        H3 = tp + 1.1 * (high - low) / 4
        L3 = tp - 1.1 * (high - low) / 4
        L4 = tp - 1.1 * (high - low) / 2
        return H3, L3, H4, L4
    
    # Shift high/low/close by 1 to use previous day's data
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Previous day's data for Camarilla calculation
    prev_high = np.concatenate([[np.nan], d_high[:-1]])
    prev_low = np.concatenate([[np.nan], d_low[:-1]])
    prev_close = np.concatenate([[np.nan], d_close[:-1]])
    
    H3, L3, H4, L4 = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Calculate ADX
    adx = calculate_adx(d_high, d_low, d_close, 14)
    
    # Align to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need ADX (28), Camarilla (1), volume MA (20)
    start_idx = max(28, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or 
            np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trend, ADX <= 25 = range
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] <= 25
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            if ranging:
                # Mean reversion in range: fade from H3/L3
                if close[i] >= H3_aligned[i] and volume_confirm:
                    signals[i] = -0.25  # short at H3
                    position = -1
                elif close[i] <= L3_aligned[i] and volume_confirm:
                    signals[i] = 0.25   # long at L3
                    position = 1
            else:  # trending
                # Breakout in trend: break through H4/L4
                if close[i] > H4_aligned[i] and volume_confirm:
                    signals[i] = 0.25   # long breakout
                    position = 1
                elif close[i] < L4_aligned[i] and volume_confirm:
                    signals[i] = -0.25  # short breakdown
                    position = -1
        elif position == 1:
            # Long exit conditions
            if ranging:
                # Exit mean reversion at opposite level or midpoint
                if close[i] <= L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # trending
                # Exit trend when price fails to hold above H4
                if close[i] < H4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        elif position == -1:
            # Short exit conditions
            if ranging:
                # Exit mean reversion at opposite level or midpoint
                if close[i] >= H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # trending
                # Exit trend when price fails to hold below L4
                if close[i] > L4_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals