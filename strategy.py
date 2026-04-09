#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# - Uses 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# - 1w EMA200 filter: only long when price > weekly EMA200, short when price < weekly EMA200
# - Volume confirmation: 6h volume > 2.0x 20-period average to ensure breakout strength
# - Discrete position sizing: 0.25 (25% of capital) to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Novelty: Combines Camarilla pivot structure with weekly trend filter to avoid counter-trend trades
# - Works in bull: breakouts at R4/S4 with trend filter; works in bear: mean reversion at R3/S3 with trend filter

name = "6h_1d_1w_camarilla_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    # Camarilla: based on previous day's range
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # fill first value
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    r4 = pivot + (range_hl * 1.1 / 2.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align 1d Camarilla levels to 6h timeframe (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Pre-compute 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or
            np.isnan(volume_spike[i]) or
            volume[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly EMA200 (trend change)
            if close[i] < ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly EMA200 (trend change)
            if close[i] > ema_200_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout/mean reversion with volume confirmation and trend filter
            # Long breakout: price breaks above R4 AND above weekly EMA200 AND volume spike
            if high[i] >= r4_aligned[i] and close[i] > ema_200_1w_aligned[i] and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short breakout: price breaks below S4 AND below weekly EMA200 AND volume spike
            elif low[i] <= s4_aligned[i] and close[i] < ema_200_1w_aligned[i] and volume_spike[i]:
                position = -1
                signals[i] = -0.25
            # Long mean reversion: price drops to S3 AND above weekly EMA200 AND volume spike
            elif low[i] <= s3_aligned[i] and close[i] > ema_200_1w_aligned[i] and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short mean reversion: price rises to R3 AND below weekly EMA200 AND volume spike
            elif high[i] >= r3_aligned[i] and close[i] < ema_200_1w_aligned[i] and volume_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals