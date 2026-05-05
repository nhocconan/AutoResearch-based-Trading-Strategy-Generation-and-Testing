#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 1w Camarilla R4 level AND price > 1d EMA34 AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below 1w Camarilla S4 level AND price < 1d EMA34 AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price crosses back below/above 1w Camarilla HLC/3 pivot point OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Weekly Camarilla R4/S4 provides stronger breakout levels than R3/S3, reducing false signals
# 1d EMA34 filters primary trend to avoid counter-trend trades
# Volume spike confirms breakout strength
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_Camarilla_R4S4_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels (based on previous weekly bar)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R4 = Pivot + Range * 1.1, S4 = Pivot - Range * 1.1
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    camarilla_r4 = pivot_1w + (range_1w * 1.1)
    camarilla_s4 = pivot_1w - (range_1w * 1.1)
    camarilla_pivot = pivot_1w  # PP level for exit (HLC/3)
    
    # Align weekly Camarilla levels to 12h timeframe (wait for completed weekly bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Camarilla R4, above 1d EMA34, volume confirmation, in session
            if close[i] > camarilla_r4_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Camarilla S4, below 1d EMA34, volume confirmation, in session
            elif close[i] < camarilla_s4_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below weekly Camarilla pivot OR volume drops below average
            if close[i] < camarilla_pivot_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above weekly Camarilla pivot OR volume drops below average
            if close[i] > camarilla_pivot_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals