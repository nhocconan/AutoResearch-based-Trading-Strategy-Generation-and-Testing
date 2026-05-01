#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h HMA21 trend filter and volume spike confirmation
# Uses Camarilla pivot levels (R3/S3) from 1d timeframe for institutional breakout levels
# 12h HMA21 filter ensures alignment with medium-term trend to avoid counter-trend whipsaws
# Volume spike (>2.0 * 20-period EMA) confirms institutional participation
# Designed for low trade frequency: ~25-40 trades/year per symbol with 0.25 sizing
# Works in bull markets via breakout continuation and bear markets via trend-following alignment
# BTC/ETH focused: requires 12h trend alignment and volume spike to avoid SOL-only bias

name = "4h_Camarilla_R3S3_12hHMA21_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for Camarilla pivots (R3/S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h HTF data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from 1d OHLC
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 12h HMA21 for trend filter
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.convolve(arr, np.arange(1, half + 1), 'valid') / (half * (half + 1) / 2)
        wma1 = np.convolve(arr, np.arange(1, period + 1), 'valid') / (period * (period + 1) / 2)
        raw = 2 * wma2 - wma1
        hma_vals = np.convolve(raw, np.arange(1, sqrt + 1), 'valid') / (sqrt * (sqrt + 1) / 2)
        # Pad to original length
        result = np.full_like(arr, np.nan)
        result[period-1:period-1+len(hma_vals)] = hma_vals
        return result
    
    close_12h = df_12h['close'].values
    hma_21_12h = hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period EMA (strict filter)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h HMA21
        bullish_bias = close[i] > hma_21_12h_aligned[i]
        bearish_bias = close[i] < hma_21_12h_aligned[i]
        
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
                signals[i] = 0.0  # Avoid chop around HMA21
        
        elif position == 1:  # Long position
            # Exit: price breaks below Camarilla S3 or price below 12h HMA21
            if close[i] < camarilla_s3_aligned[i] or close[i] < hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Camarilla R3 or price above 12h HMA21
            if close[i] > camarilla_r3_aligned[i] or close[i] > hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals