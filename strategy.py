#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA200 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND price > 1d EMA200 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S3 AND price < 1d EMA200 AND volume > 2.0 * avg_volume(20)
# Exit when price crosses Camarilla H4/L4 (extreme) OR volume < avg_volume(20)
# Uses discrete sizing 0.25 to minimize fee churn
# Target: 75-200 total trades over 4 years (19-50/year)
# Camarilla levels from 1d provide robust daily support/resistance; 1d EMA200 filters long-term trend; volume spike confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)

name = "4h_Camarilla_R3S3_Breakout_1dEMA200_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for Camarilla calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d / 4  # R3
    camarilla_l3 = close_1d - 1.1 * range_1d / 4  # S3
    camarilla_h4 = close_1d + 1.1 * range_1d / 2  # R4
    camarilla_l4 = close_1d - 1.1 * range_1d / 2  # S4
    
    # Align Camarilla levels to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 1d EMA200
    close_1d_series = pd.Series(close_1d)
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3, above 1d EMA200, volume confirmation
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema200_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3, below 1d EMA200, volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema200_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Camarilla L4 (extreme support) OR volume drops below average
            if close[i] < camarilla_l4_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Camarilla H4 (extreme resistance) OR volume drops below average
            if close[i] > camarilla_h4_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals