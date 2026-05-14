#!/usr/bin/env python3
"""
4h Camarilla R3/S3 Breakout + 1d HMA21 Trend + Volume Spike with ATR Stoploss
Hypothesis: Camarilla R3/S3 levels represent stronger institutional support/resistance than R1/S1.
Breakouts above R3 or below S3 with daily HMA21 trend alignment and volume spike capture
strong institutional moves. ATR-based stoploss limits drawdown. Works in bull markets (trend
continuation) and bear markets (mean reversion to pivot levels). 4h timeframe targets
20-50 trades/year to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(series, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.concatenate([np.full(half_period-1, np.nan), wma_half])
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(series, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.concatenate([np.full(period-1, np.nan), wma_full])
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    wma_sqrt = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    wma_sqrt = np.concatenate([np.full(sqrt_period-1, np.nan), wma_sqrt])
    
    return wma_sqrt

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Daily data for Camarilla pivots and HMA21 (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for daily data
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    daily_range = daily_high - daily_low
    camarilla_r3 = daily_close + 1.1 * daily_range
    camarilla_s3 = daily_close - 1.1 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily HMA21 trend filter
    hma_21_1d = calculate_hma(daily_close, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for daily data and volume MA
    start_idx = max(34, 20, 14) + 5  # extra for safety
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions
        breakout_long = curr_close > camarilla_r3_aligned[i]
        breakout_short = curr_close < camarilla_s3_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Camarilla breakout + volume spike + daily HMA21 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > hma_21_1d_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < hma_21_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on retrace to S3, trend change, or ATR stoploss
            stoploss_level = entry_price - 2.5 * atr[i]
            if curr_close < camarilla_s3_aligned[i] or curr_close < hma_21_1d_aligned[i] or curr_close < stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on retrace to R3, trend change, or ATR stoploss
            stoploss_level = entry_price + 2.5 * atr[i]
            if curr_close > camarilla_r3_aligned[i] or curr_close > hma_21_1d_aligned[i] or curr_close > stoploss_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dHMA21_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0