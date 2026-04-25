#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_ChopRegime
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and choppiness regime filter.
Only trade breakouts in direction of daily trend when market is trending (CHOP < 38.2).
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
Designed to work in both bull and bear markets via trend alignment and regime filtering.
Camarilla levels provide institutional support/resistance with high breakout reliability.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF trend filter, Camarilla calculation, and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels on 1d data
    # Camarilla: R3 = C + ((H-L)*1.1/4), S3 = C - ((H-L)*1.1/4)
    camarilla_r3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4)
    camarilla_s3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    
    # Calculate choppiness index on 1d data (CHOP > 61.8 = range, CHOP < 38.2 = trend)
    def calculate_choppiness(high_arr, low_arr, close_arr, period=14):
        """Calculate choppiness index"""
        atr_sum = np.zeros(len(close_arr))
        for i in range(period, len(close_arr)):
            atr_sum[i] = atr_sum[i-1] + np.max([
                high_arr[i] - low_arr[i],
                abs(high_arr[i] - close_arr[i-1]),
                abs(low_arr[i] - close_arr[i-1])
            ])
        
        atr = np.zeros(len(close_arr))
        atr[period:] = atr_sum[period:] / period
        
        chop = np.zeros(len(close_arr))
        for i in range(period, len(close_arr)):
            highest_high = np.max(high_arr[i-period+1:i+1])
            lowest_low = np.min(low_arr[i-period+1:i+1])
            if highest_high != lowest_low:
                chop[i] = 100 * np.log10(atr_sum[i] / (period * (highest_high - lowest_low))) / np.log10(period)
            else:
                chop[i] = 50  # neutral when no range
        return chop
    
    chop_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Align HTF indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d, additional_delay_bars=1)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d, additional_delay_bars=1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 and chop (50)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade when market is trending (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Look for Camarilla breakout signals with trend filter
            # Long: price breaks above R3 in uptrend (close > EMA50)
            # Short: price breaks below S3 in downtrend (close < EMA50)
            long_signal = (close[i] > camarilla_r3_aligned[i]) and (close[i] > ema50_aligned[i])
            short_signal = (close[i] < camarilla_s3_aligned[i]) and (close[i] < ema50_aligned[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA50 (trend reversal) OR chop increases (range market)
            exit_signal = close[i] < ema50_aligned[i] or chop_aligned[i] >= 38.2
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal) OR chop increases (range market)
            exit_signal = close[i] > ema50_aligned[i] or chop_aligned[i] >= 38.2
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_ChopRegime"
timeframe = "12h"
leverage = 1.0