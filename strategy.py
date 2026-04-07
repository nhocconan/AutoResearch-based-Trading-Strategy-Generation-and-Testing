#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + Weekly Trend + Volume Spike
# Hypothesis: Camarilla pivot levels (R3/S3 for reversal, R4/S4 for breakout) combined
# with weekly trend filter and volume confirmation captures institutional
# turning points and breakouts. Works in bull/bear by fading extremes in range
# and following breakouts in trends. Target: 50-150 total trades over 4 years.

name = "6h_camarilla_pivot_weekly_trend_volume_v1"
timeframe = "6h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical Price = (High + Low + Close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot = Typical Price
    pivot = typical_price.values
    # R1 = Close + 1.1*(High - Low)
    r1 = df_1d['close'].values + 1.1 * (df_1d['high'] - df_1d['low']).values
    # S1 = Close - 1.1*(High - Low)
    s1 = df_1d['close'].values - 1.1 * (df_1d['high'] - df_1d['low']).values
    # R2 = Pivot + 1.1*(High - Low)
    r2 = pivot + 1.1 * (df_1d['high'] - df_1d['low']).values
    # S2 = Pivot - 1.1*(High - Low)
    s2 = pivot - 1.1 * (df_1d['high'] - df_1d['low']).values
    # R3 = High + 2*(High - Low)
    r3 = df_1d['high'].values + 2 * (df_1d['high'] - df_1d['low']).values
    # S3 = Low - 2*(High - Low)
    s3 = df_1d['low'].values - 2 * (df_1d['high'] - df_1d['low']).values
    # R4 = High + 2.6*(High - Low)
    r4 = df_1d['high'].values + 2.6 * (df_1d['high'] - df_1d['low']).values
    # S4 = Low - 2.6*(High - Low)
    s4 = df_1d['low'].values - 2.6 * (df_1d['high'] - df_1d['low']).values
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or weekly trend turns bearish
            if close[i] < s3_6h[i] or close[i] < ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or weekly trend turns bullish
            if close[i] > r3_6h[i] or close[i] > ema_20_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Fade at R3/S3 in ranging markets (price rejects extreme levels)
                if close[i] < r3_6h[i] and close[i] > s3_6h[i]:
                    # Near R3: look for rejection
                    if close[i] > r3_6h[i] * 0.998 and close[i] < r3_6h[i]:  # Within 0.2% of R3
                        # Check if weekly trend is not strongly bullish (avoid fighting trend)
                        if close[i] <= ema_20_1w_aligned[i] * 1.02:  # Not far above weekly EMA
                            position = -1
                            signals[i] = -0.25
                    # Near S3: look for rejection
                    elif close[i] < s3_6h[i] * 1.002 and close[i] > s3_6h[i]:  # Within 0.2% of S3
                        # Check if weekly trend is not strongly bearish
                        if close[i] >= ema_20_1w_aligned[i] * 0.98:  # Not far below weekly EMA
                            position = 1
                            signals[i] = 0.25
                # Breakout at R4/S4 in trending markets
                elif close[i] > r4_6h[i] and close[i] > ema_20_1w_aligned[i]:
                    # Breakout above R4 with bullish weekly trend
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_6h[i] and close[i] < ema_20_1w_aligned[i]:
                    # Breakdown below S4 with bearish weekly trend
                    position = -1
                    signals[i] = -0.25
    
    return signals