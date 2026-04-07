#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + 1d Trend + Volume Spike
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) from daily timeframe identify key support/resistance.
# In trending markets (1d EMA50), breaks of R4/S4 with volume continuation signal strong momentum.
# In ranging markets, reversals at R3/S3 with volume exhaustion capture mean reversion.
# Works in bull/bear via trend filter; volume confirms institutional participation.
# Target: 12-37 trades/year (50-150 total over 4 years) for 6h timeframe.

name = "6h_camarilla_pivot_1d_trend_volume_v1"
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
    
    # Get 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Close + Range * 1.1/2
    # R3 = Close + Range * 1.1/4
    # S3 = Close - Range * 1.1/4
    # S4 = Close - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 6s timeframe (shifted by 1 for completed bar)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced for signal frequency)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion fail) or trend turns bearish
            if close[i] < s3_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion fail) or trend turns bullish
            if close[i] > r3_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout mode: price breaks R4/S4 with volume in trending market
                if close[i] > r4_1d_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < s4_1d_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
                # Mean reversion mode: price reverses from R3/S3 in ranging market
                elif close[i] < r3_1d_aligned[i] and close[i] > s3_1d_aligned[i]:
                    # In range, look for rejection at levels
                    if i >= 2:
                        # Rejection at R3: bearish close after touching/resisting R3
                        if (high[i] >= r3_1d_aligned[i] * 0.999 and close[i] < r3_1d_aligned[i] and 
                            close[i] < close[i-1]):
                            position = -1
                            signals[i] = -0.25
                        # Rejection at S3: bullish close after touching/supporting S3
                        elif (low[i] <= s3_1d_aligned[i] * 1.001 and close[i] > s3_1d_aligned[i] and 
                              close[i] > close[i-1]):
                            position = 1
                            signals[i] = 0.25
    
    return signals