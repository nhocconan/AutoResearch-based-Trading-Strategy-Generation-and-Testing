#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3, S3) from 12h data for institutional breakout levels
# 12h EMA50 filter ensures alignment with medium-term trend to avoid counter-trend whipsaws
# Volume spike (>2.0 * 20-period EMA) confirms institutional participation
# Designed for low trade frequency: ~12-30 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# BTC/ETH focused: requires 12h trend alignment and volume spike to avoid SOL-only bias

name = "6h_Camarilla_R3S3_12hEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for Camarilla pivot calculation (using previous 12h bar)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    camarilla_r3 = close_12h + (range_12h * 1.1 / 2.0)
    camarilla_s3 = close_12h - (range_12h * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (use previous 12h bar's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # 12h HTF data for EMA50 trend filter
    close_12h_vals = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (strict filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: price breaks above Camarilla R3 with volume spike
                if close[i] > camarilla_r3_aligned[i-1] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: price breaks below Camarilla S3 with volume spike
                if close[i] < camarilla_s3_aligned[i-1] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around EMA50
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or price below 12h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or price above 12h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals