#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6H Weekly Camarilla Pivot with Volume and ATR Filter
# Hypothesis: Camarilla pivot levels (R3/S3, R4/S4) on weekly timeframe act as strong
# support/resistance zones. Price rejecting R3/S3 with volume confirmation indicates
# reversal, while breaking R4/S4 with volume indicates continuation. ATR filter avoids
# choppy markets. Works in bull/bear by fading extremes and catching breakouts.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "6h_weekly_camarilla_pivot_volume_atr_v1"
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
    
    # Get weekly data for Camarilla pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly OHLC for pivot
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots using previous week's data
    # Shift by 1 to use only completed weekly bars
    prev_weekly_high = np.concatenate([[np.nan], weekly_high[:-1]])
    prev_weekly_low = np.concatenate([[np.nan], weekly_low[:-1]])
    prev_weekly_close = np.concatenate([[np.nan], weekly_close[:-1]])
    
    # Weekly pivot point (PP)
    pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    
    # Weekly range
    weekly_range = prev_weekly_high - prev_weekly_low
    
    # Camarilla levels
    r4 = pp + (weekly_range * 1.1 / 2)
    r3 = pp + (weekly_range * 1.1 / 4)
    s3 = pp - (weekly_range * 1.1 / 4)
    s4 = pp - (weekly_range * 1.1 / 2)
    
    # ATR filter on 6H to avoid choppy markets
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[1:atr_period])
        for i in range(atr_period, len(tr)):
            if not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Normalized ATR (ATR as % of price)
    atr_pct = atr / close
    
    # Volume filter: volume > 1.3x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.3 * vol_ma)
    
    # Align weekly Camarilla levels to 6H timeframe
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    atr_pct_aligned = align_htf_to_ltf(prices, df_weekly, atr_pct)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_pct_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Avoid extremely choppy markets (high ATR %)
        if atr_pct_aligned[i] > 0.08:  # 8% ATR threshold
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 or ATR expands significantly
            if close[i] < s3_aligned[i] or atr_pct_aligned[i] > 0.06:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price breaks above R3 or ATR expands significantly
            if close[i] > r3_aligned[i] or atr_pct_aligned[i] > 0.06:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Fade at R3/S3 with volume confirmation (mean reversion)
            if (close[i] <= r3_aligned[i] and low[i] < r3_aligned[i] and vol_filter[i]):
                # Potential short at R3 rejection
                position = -1
                signals[i] = -0.25
            elif (close[i] >= s3_aligned[i] and high[i] > s3_aligned[i] and vol_filter[i]):
                # Potential long at S3 rejection
                position = 1
                signals[i] = 0.25
            # Breakout at R4/S4 with volume confirmation (trend continuation)
            elif (high[i] > r4_aligned[i] and close[i] > r4_aligned[i] and vol_filter[i]):
                # Breakout above R4
                position = 1
                signals[i] = 0.25
            elif (low[i] < s4_aligned[i] and close[i] < s4_aligned[i] and vol_filter[i]):
                # Breakdown below S4
                position = -1
                signals[i] = -0.25
    
    return signals