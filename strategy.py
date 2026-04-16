#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above Camarilla R3 (1d) + 1d volume > 1.5x 20-period avg + CHOP(14) < 61.8 (trending regime)
# Short when price breaks below Camarilla S3 (1d) + 1d volume > 1.5x 20-period avg + CHOP(14) < 61.8 (trending regime)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla levels provide precise intraday support/resistance that work in ranging and trending markets.
# Volume threshold (1.5x) targets ~20-30 trades/year on 12h timeframe to avoid overtrading.
# CHOP filter avoids whipsaws in ranging markets, only allowing trades in trending conditions.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivots (R3, S3) and CHOP ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2.0
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2.0
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Choppiness Index (CHOP) - requires high, low, close
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index"""
        atr = np.zeros(len(close_arr))
        for i in range(period, len(close_arr)):
            tr = np.max([
                high_arr[i] - low_arr[i],
                np.abs(high_arr[i] - close_arr[i-1]),
                np.abs(low_arr[i] - close_arr[i-1])
            ])
            atr[i] = tr
        # Smooth ATR with Wilder's smoothing (equivalent to RMA)
        atr_smoothed = np.zeros(len(close_arr))
        if period < len(close_arr):
            atr_smoothed[period] = np.mean(atr[1:period+1])
            for i in range(period+1, len(close_arr)):
                atr_smoothed[i] = (atr_smoothed[i-1] * (period-1) + atr[i]) / period
        
        # Calculate CHOP: 100 * log10(sum(ATR) / (max(high)-min(low))) / log10(period)
        chop = np.full(len(close_arr), 50.0)  # default to neutral
        for i in range(period, len(close_arr)):
            if i >= period:
                atr_sum = np.sum(atr_smoothed[i-period+1:i+1])
                max_high = np.max(high_arr[i-period+1:i+1])
                min_low = np.min(low_arr[i-period+1:i+1])
                if max_high > min_low and atr_sum > 0:
                    chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(period)
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Volume SMA ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Camarilla (20), CHOP (14), volume (20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Regime filter: CHOP < 61.8 (trending market)
        trending_regime = chop_1d_aligned[i] < 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (close > R3)
        # 2. Volume confirmation
        # 3. Trending regime
        if (close[i] > camarilla_r3_1d_aligned[i]) and \
           vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (close < S3)
        # 2. Volume confirmation
        # 3. Trending regime
        elif (close[i] < camarilla_s3_1d_aligned[i]) and \
             vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolume_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0