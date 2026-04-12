#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h_1d_vortex_v1
# Vortex indicator (VI+) and (VI-) identifies trend direction. 
# Long when VI+ crosses above VI- in uptrend, short when VI- crosses above VI+ in downtrend.
# Uses 1d timeframe for Vortex calculation to reduce noise. 
# Includes volume confirmation (volume > 20-period average) and volatility filter (ATR-based).
# Designed for low trade frequency (<30/year) with strong trend signals.
# Works in both bull and bear markets by following major trends.
name = "12h_1d_vortex_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Vortex and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Calculate +DM and -DM for 1d
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (alpha = 1/period)
    period = 14
    alpha = 1.0 / period
    
    def wilders_smoothing(data, alpha):
        result = np.full_like(data, np.nan)
        for i in range(len(data)):
            if np.isnan(data[i]):
                if i == 0:
                    result[i] = np.nan
                else:
                    result[i] = result[i-1]
            else:
                if i == 0 or np.isnan(result[i-1]):
                    result[i] = data[i]
                else:
                    result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
        return result
    
    tr_smooth = wilders_smoothing(tr, alpha)
    dm_plus_smooth = wilders_smoothing(dm_plus, alpha)
    dm_minus_smooth = wilders_smoothing(dm_minus, alpha)
    
    # Calculate VI+ and VI-
    vi_plus = dm_plus_smooth / tr_smooth
    vi_minus = dm_minus_smooth / tr_smooth
    
    # Calculate ATR for volatility filter (14-period ATR)
    atr = tr_smooth  # Wilder's ATR is the smoothed TR
    
    # Calculate 20-period average volume for volume filter
    vol_avg = np.convolve(volume_1d, np.ones(20)/20, mode='same')
    vol_avg[:10] = np.nan
    vol_avg[-10:] = np.nan
    
    # Align indicators to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if indicators not ready
        if np.isnan(vi_plus_aligned[i]) or np.isnan(vi_minus_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(vol_avg_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 20-period average
        volume_filter = volume[i] > vol_avg_aligned[i]
        
        # Volatility filter: avoid extremely low volatility (ATR < 50% of its 50-period average)
        if i >= 50:
            atr_ma = np.nanmean(atr_aligned[i-50:i])
            vol_filter = atr_aligned[i] > 0.5 * atr_ma if not np.isnan(atr_ma) else True
        else:
            vol_filter = True
        
        # Vortex crossover signals
        vi_plus_cross_above = vi_plus_aligned[i] > vi_minus_aligned[i] and vi_plus_aligned[i-1] <= vi_minus_aligned[i-1]
        vi_minus_cross_above = vi_minus_aligned[i] > vi_plus_aligned[i] and vi_minus_aligned[i-1] <= vi_plus_aligned[i-1]
        
        # Trend strength: only take signal if VI+ or VI- is significantly above the other
        trend_filter = np.abs(vi_plus_aligned[i] - vi_minus_aligned[i]) > 0.1
        
        # Long signal: VI+ crosses above VI- with volume and volatility confirmation
        long_signal = vi_plus_cross_above and volume_filter and vol_filter and trend_filter
        
        # Short signal: VI- crosses above VI+ with volume and volatility confirmation
        short_signal = vi_minus_cross_above and volume_filter and vol_filter and trend_filter
        
        # Exit on opposite crossover
        exit_long = vi_minus_cross_above
        exit_short = vi_plus_cross_above
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals