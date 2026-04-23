#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla Pivot Breakout with 1w EMA Trend Filter and Volume Spike.
Long when price breaks above Camarilla R3 (1d) AND 1w EMA50 is rising AND volume > 2.0x 20-period average.
Short when price breaks below Camarilla S3 (1d) AND 1w EMA50 is falling AND volume > 2.0x 20-period average.
Exit when price returns to Camarilla Pivot Point (1d) or opposite Camarilla level (S3 for longs, R3 for shorts).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Camarilla levels provide precise intraday support/resistance, while 1w EMA ensures alignment with weekly trend.
Volume confirmation filters weak breakouts. Designed to work in both bull and bear markets by requiring trend alignment.
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
    open_prices = prices['open'].values
    
    # Load 1d data for Camarilla levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Camarilla R3 = close_1d + (high_1d - low_1d) * 1.1/4
    # Camarilla S3 = close_1d - (high_1d - low_1d) * 1.1/4
    # Camarilla R4 = close_1d + (high_1d - low_1d) * 1.1/2
    # Camarilla S4 = close_1d - (high_1d - low_1d) * 1.1/2
    # Pivot Point = (high_1d + low_1d + close_1d) / 3
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    camarilla_pp = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    camarilla_range = prev_high_1d - prev_low_1d
    camarilla_r3 = prev_close_1d + camarilla_range * 1.1 / 4.0
    camarilla_s3 = prev_close_1d - camarilla_range * 1.1 / 4.0
    camarilla_r4 = prev_close_1d + camarilla_range * 1.1 / 2.0
    camarilla_s4 = prev_close_1d - camarilla_range * 1.1 / 2.0
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA50 slope (rising/falling) - use 3-period difference
    ema50_slope = np.zeros_like(ema50_1w)
    ema50_slope[3:] = ema50_1w[3:] - ema50_1w[:-3]
    
    # Align HTF indicators to 1d timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Ensure warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema50_slope_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1w EMA50 rising AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                ema50_slope_aligned[i] > 0 and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 1w EMA50 falling AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  ema50_slope_aligned[i] < 0 and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price returns to Camarilla Pivot Point
            if position == 1 and price <= camarilla_pp_aligned[i]:
                exit_signal = True
            elif position == -1 and price >= camarilla_pp_aligned[i]:
                exit_signal = True
            
            # Alternative exit: Price reaches opposite Camarilla level (S3 for longs, R3 for shorts)
            elif position == 1 and price >= camarilla_s3_aligned[i]:
                exit_signal = True
            elif position == -1 and price <= camarilla_r3_aligned[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0