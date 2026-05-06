#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 pivot breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above 4h Camarilla R3 level AND 1d close > 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Short when price breaks below 4h Camarilla S3 level AND 1d close < 1d EMA34 AND volume > 2.0 * 20-bar average volume
# Exit when price retests the 4h Camarilla pivot point (mean of H+L+C from prior day)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla pivots provide intraday support/resistance levels that work in both bull and bear markets
# 1d EMA34 filters for higher timeframe trend alignment (proven edge from research)
# Volume spike confirmation reduces false breakouts during low participation

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla levels and 1d EMA34 ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 1 or len(df_1d) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (based on prior 1d bar)
    # Camarilla: Pivot = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need prior day's HLC, so shift by 1
    prior_high = np.roll(high_4h, 1)
    prior_low = np.roll(low_4h, 1)
    prior_close = np.roll(close_4h, 1)
    # First value will be invalid due to roll, handled by min_periods in alignment
    pivot = (prior_high + prior_low + prior_close) / 3.0
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 2.0
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 2.0
    
    # Calculate 1d EMA34 trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 4h timeframe (wait for completed bars)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0 * 20-bar average volume (spike filter)
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > R3 AND uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < S3 AND downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests pivot from above
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests pivot from below
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals