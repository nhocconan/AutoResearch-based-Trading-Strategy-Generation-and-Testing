#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R4/S4 Breakout with 1d trend filter and volume confirmation
# Uses 1d Camarilla levels (more reliable than 4h) to avoid whipsaw. Breaks above R4 or below S4
# indicate strong momentum. 1d EMA50 filter ensures we trade with higher timeframe trend.
# Volume spike (>2x average) confirms institutional participation. Target: 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H-L = range
    # R4 = Close + (1.1/2) * (High - Low) = Close + 0.55 * (High - Low)
    # S4 = Close - (1.1/2) * (High - Low) = Close - 0.55 * (High - Low)
    # These are the strongest breakout levels
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 0.55 * range_1d
    camarilla_s4 = close_1d - 0.55 * range_1d
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA50 for trend filter (must be calculated on completed 1d bars)
    close_1d_series = pd.Series(close_1d.values)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2x 24-period average (6 hours of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long: Break above R4 (strong bullish breakout) + uptrend + volume
        if (close[i] > camarilla_r4_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and   # Uptrend filter
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short: Break below S4 (strong bearish breakdown) + downtrend + volume
        elif (close[i] < camarilla_s4_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and   # Downtrend filter
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0