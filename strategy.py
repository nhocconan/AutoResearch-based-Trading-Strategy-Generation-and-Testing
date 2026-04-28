#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla R4/S4 levels with 4h HMA21 trend filter and volume confirmation.
# Enter long when price breaks above 1d Camarilla R4 level with volume > 1.8x average and close > 4h HMA21 (bullish bias).
# Enter short when price breaks below 1d Camarilla S4 level with volume > 1.8x average and close < 4h HMA21 (bearish bias).
# Exit when price returns to the 1d Camarilla midpoint (P) or touches the opposite level (S4 for long exit, R4 for short exit).
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 75-200 total trades over 4 years.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# R4/S4 levels are stronger breakout levels than R3/S3, reducing false signals and trade frequency.

name = "4h_Camarilla_R4S4_Breakout_4hHMA21_VolumeConfirm_v1"
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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for Camarilla calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Camarilla levels (based on previous day's close and range)
    camarilla_pivot = close_1d  # Pivot is previous close
    camarilla_range = high_1d - low_1d
    
    # R4 and S4 levels (stronger breakout levels for fewer trades)
    r4 = camarilla_pivot + camarilla_range * 1.1 / 2
    s4 = camarilla_pivot - camarilla_range * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 4h data for HMA21 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Calculate 4h HMA21
    close_4h = df_4h['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='same')
    
    wma_half = wma(close_4h, half_len)
    wma_full = wma(close_4h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21_4h = wma(raw_hma, sqrt_len)
    hma_21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_21_4h)
    
    # Calculate volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(hma_21_4h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 4h HMA21 bias
        bullish_bias = close[i] > hma_21_4h_aligned[i]
        bearish_bias = close[i] < hma_21_4h_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r4_aligned[i]
        short_breakout = close[i] < s4_aligned[i]
        
        # Exit conditions: return to pivot or touch opposite level
        long_exit = close[i] < pivot_aligned[i]
        short_exit = close[i] > pivot_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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