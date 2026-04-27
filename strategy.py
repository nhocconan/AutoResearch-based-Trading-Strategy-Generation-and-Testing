#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Camarilla R3/S3 breakouts on 6h with weekly trend filter and volume confirmation.
- Weekly trend filter (price vs weekly EMA34) ensures trades align with higher timeframe direction.
- Camarilla R3/S3 levels provide institutional support/resistance for breakouts.
- Volume confirmation filters weak breakouts.
- Works in bull (breakouts continue with weekly uptrend) and bear (false breakouts reversed via weekly downtrend).
- Target: 15-30 trades/year to minimize fee drag on 6s timeframe.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3, S3, R4, S4
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    camarilla_r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    camarilla_s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.8 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA, daily Camarilla, and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema34_1w_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        r4 = camarilla_r4_aligned[i]
        s4 = camarilla_s4_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume spike and weekly uptrend
            if close[i] > r3 and vol_spike_val and close[i] > weekly_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume spike and weekly downtrend
            elif close[i] < s3 and vol_spike_val and close[i] < weekly_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S3 or weekly trend turns down
            if close[i] < s3 or close[i] < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 or weekly trend turns up
            if close[i] > r3 or close[i] > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0