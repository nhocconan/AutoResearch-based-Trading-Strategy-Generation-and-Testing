#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume confirmation and 1d HMA trend filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2.0x 24-period volume SMA + price > 1d HMA21
# Short when price breaks below Camarilla S3 (1d) + volume > 2.0x 24-period volume SMA + price < 1d HMA21
# Uses tighter Camarilla levels (R3/S3) for stronger breakouts, 1d HMA for trend alignment, and strict volume filter
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing significant intraday moves

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) and HMA21 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_r3 = camarilla_pivot + (range_1d * 1.1 / 4)  # R3 level
    camarilla_s3 = camarilla_pivot - (range_1d * 1.1 / 4)  # S3 level
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # HMA21 calculation: Weighted Moving Average of WMA
    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        # Need to align lengths: 2*WMA(half) - WMA(full)
        # Then WMA of that with sqrt_period
        if len(wma_half) < 1 or len(wma_full) < 1:
            return np.full_like(arr, np.nan)
        # Pad to original length
        wma_half_padded = np.full_like(arr, np.nan)
        wma_full_padded = np.full_like(arr, np.nan)
        wma_half_padded[half_period-1:] = wma_half
        wma_full_padded[period-1:] = wma_full
        diff = 2 * wma_half_padded - wma_full_padded
        hma_vals = wma(diff, sqrt_period)
        # Pad result
        hma_result = np.full_like(arr, np.nan)
        hma_result[period-1:] = hma_vals
        return hma_result
    
    hma_21_1d = hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 24-period volume SMA
        vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation
        # 3. Price above 1d HMA21 (uptrend filter)
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm and (close[i] > hma_21_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation
        # 3. Price below 1d HMA21 (downtrend filter)
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm and (close[i] < hma_21_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_Volume_HMA21_Filter_v1"
timeframe = "12h"
leverage = 1.0