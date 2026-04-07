#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla Pivot + Daily Volume Spike + 1w Trend
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# In ranging markets, price reverts to mean near these levels. In trending markets,
# breaks above/below key levels (L4, H4) with volume continuation signal strong moves.
# Weekly trend filter ensures we only trade in direction of higher timeframe momentum.
# Works in bull markets via buying dips at S3/S4 in uptrend, selling rallies at R3/R4.
# Works in bear via selling rallies at R3/R4 in downtrend, buying dips at S3/S4.
# Volume spike confirms institutional participation at key levels.
# Target: 20-50 trades/year (80-200 total over 4 years) for 4h timeframe.

name = "4h_camarilla_pivot_daily_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # Typical price = (H + L + C) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H4 = Close + 1.1 * (High - Low) / 2
    # L4 = Close - 1.1 * (High - Low) / 2
    # H3 = Close + 1.1 * (High - Low) / 4
    # L3 = Close - 1.1 * (High - Low) / 4
    # H2 = Close + 1.1 * (High - Low) / 6
    # L2 = Close - 1.1 * (High - Low) / 6
    # H1 = Close + 1.1 * (High - Low) / 12
    # L1 = Close - 1.1 * (High - Low) / 12
    
    H4 = close_1d + 1.1 * range_1d / 2.0
    L4 = close_1d - 1.1 * range_1d / 2.0
    H3 = close_1d + 1.1 * range_1d / 4.0
    L3 = close_1d - 1.1 * range_1d / 4.0
    H2 = close_1d + 1.1 * range_1d / 6.0
    L2 = close_1d - 1.1 * range_1d / 6.0
    H1 = close_1d + 1.1 * range_1d / 12.0
    L1 = close_1d - 1.1 * range_1d / 12.0
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below L3 or weekly trend turns bearish
            if close[i] < L3_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above H3 or weekly trend turns bullish
            if close[i] > H3_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long setup: price near S3/S4 with weekly uptrend
                if (close[i] <= L3_aligned[i] * 1.005 or close[i] <= L4_aligned[i] * 1.005) and close[i] > ema_20_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short setup: price near R3/R4 with weekly downtrend
                elif (close[i] >= H3_aligned[i] * 0.995 or close[i] >= H4_aligned[i] * 0.995) and close[i] < ema_20_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals