#!/usr/bin/env python3
"""
4H_ParabolicSAR_VolumeSpike_12H_TrendFilter_v1
Hypothesis: Use Parabolic SAR (0.02, 0.2) on 4h for trend direction and reversal signals.
Enter long when SAR flips below price (bullish reversal) with volume confirmation (>1.5x 20-bar avg) and 12h EMA50 uptrend.
Enter short when SAR flips above price (bearish reversal) with volume confirmation and 12h EMA50 downtrend.
Exit when SAR flips back (trend reversal). This captures trend reversals with volume confirmation and higher-timeframe trend filter to avoid false signals in chop.
Designed to work in both bull (catching uptrend reversals) and bear (catching downtrend reversals) markets.
"""
name = "4H_ParabolicSAR_VolumeSpike_12H_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Parabolic SAR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Parabolic SAR on 4h
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Initialize SAR
    sar = np.zeros_like(close_4h)
    trend = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    af = 0.02  # acceleration factor
    max_af = 0.2
    ep = 0.0   # extreme point
    
    # Set initial values
    sar[0] = low_4h[0]
    trend[0] = 1
    ep = high_4h[0]
    
    for i in range(1, len(close_4h)):
        if trend[i-1] == 1:  # was uptrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            # Check for trend reversal
            if low_4h[i] < sar[i]:
                trend[i] = -1
                sar[i] = ep
                ep = low_4h[i]
                af = 0.02
            else:
                trend[i] = 1
                if high_4h[i] > ep:
                    ep = high_4h[i]
                    af = min(af + 0.02, max_af)
        else:  # was downtrend
            sar[i] = sar[i-1] + af * (ep - sar[i-1])
            # Check for trend reversal
            if high_4h[i] > sar[i]:
                trend[i] = 1
                sar[i] = ep
                ep = high_4h[i]
                af = 0.02
            else:
                trend[i] = -1
                if low_4h[i] < ep:
                    ep = low_4h[i]
                    af = min(af + 0.02, max_af)
    
    sar_aligned = align_htf_to_ltf(prices, df_4h, sar)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(10, 20, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(sar_aligned[i]) or np.isnan(trend_4h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (1.3 days on 4h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: SAR flips below price (bullish reversal) with volume and 12h EMA uptrend
            if (close[i] > sar_aligned[i] and close[i-1] <= sar_aligned[i-1] and 
                volume_filter[i] and ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: SAR flips above price (bearish reversal) with volume and 12h EMA downtrend
            elif (close[i] < sar_aligned[i] and close[i-1] >= sar_aligned[i-1] and 
                  volume_filter[i] and ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: SAR flips back (trend reversal)
            if position == 1 and close[i] < sar_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > sar_aligned[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals