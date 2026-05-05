#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Camarilla pivot breakout with 4h HMA21 trend filter and volume spike confirmation
# Long when price breaks above daily Camarilla R3 level AND price > 4h HMA21 AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below daily Camarilla S3 level AND price < 4h HMA21 AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price crosses back below/above daily Camarilla pivot point OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Daily Camarilla provides robust support/resistance from higher timeframe
# 4h HMA21 filters primary trend to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_Camarilla_R3S3_Breakout_4hHMA21_VolumeSpike"
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
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (based on previous daily bar)
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = Pivot + Range * 1.1/2, S3 = Pivot - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r3 = pivot_1d + (range_1d * 1.1 / 2.0)
    camarilla_s3 = pivot_1d - (range_1d * 1.1 / 2.0)
    camarilla_pivot = pivot_1d  # PP level for exit
    
    # Align daily Camarilla levels to 4h timeframe (wait for completed daily bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Get 4h data ONCE before loop for HMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:  # Need enough for HMA21
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA21: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half = window // 2
        sqrt_n = int(np.sqrt(window))
        wma_half = wma(values, half)
        wma_full = wma(values, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_n)
    
    hma21_4h = hma(close_4h, 21)
    hma21_4h_aligned = align_htf_to_ltf(prices, df_4h, hma21_4h)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(hma21_4h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Camarilla R3, above 4h HMA21, volume confirmation, in session
            if close[i] > camarilla_r3_aligned[i] and close[i] > hma21_4h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Camarilla S3, below 4h HMA21, volume confirmation, in session
            elif close[i] < camarilla_s3_aligned[i] and close[i] < hma21_4h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below daily Camarilla pivot OR volume drops below average
            if close[i] < camarilla_pivot_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above daily Camarilla pivot OR volume drops below average
            if close[i] > camarilla_pivot_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals