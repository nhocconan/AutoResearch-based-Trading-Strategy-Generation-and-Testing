#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_1dTrend_Volume
Hypothesis: Price breaking Camarilla R3/S3 levels on 12h with 1-day EMA200 trend filter and volume confirmation captures institutional breakout moves. Works in bull (breakouts above R3 in uptrend) and bear (breakdowns below S3 in downtrend). Low-frequency signals via 12h timeframe and confluence of price level, trend, and volume.
"""
name = "12h_Camarilla_Pivot_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Calculate Camarilla levels using previous 12h bar
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2 where C=(H+L+Close)/3
    # Use previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    camarilla_r3 = pivot + (range_hl * 1.1 / 2)
    camarilla_s3 = pivot - (range_hl * 1.1 / 2)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + 1d uptrend + volume
            if close[i] > camarilla_r3[i] and close[i] > ema_200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + 1d downtrend + volume
            elif close[i] < camarilla_s3[i] and close[i] < ema_200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to pivot level (mean reversion)
            if position == 1:
                if close[i] <= pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals