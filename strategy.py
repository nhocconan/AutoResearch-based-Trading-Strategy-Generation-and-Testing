#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Camarilla Pivot with Volume and Trend Filter
# Hypothesis: Price reacting at weekly Camarilla pivot levels (S3/R3 for reversal, S4/R4 for breakout)
# with volume confirmation and trend filter (price vs 200 EMA) works in both bull and bear markets.
# In bull markets: buy at S3 reversals, sell at R4 breakouts.
# In bear markets: sell at R3 reversals, buy at S4 breakouts.
# Target: 8-20 trades/year (32-80 over 4 years).

name = "1d_weekly_camarilla_pivot_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels based on previous week's OHLC
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    range_hl = weekly_high - weekly_low
    
    # Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    pivot = np.roll(pivot, 1)
    r4 = np.roll(r4, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    s4 = np.roll(s4, 1)
    
    # Handle first element
    if len(pivot) > 1:
        pivot[0] = pivot[1]
        r4[0] = r4[1]
        r3[0] = r3[1]
        s3[0] = s3[1]
        s4[0] = s4[1]
    else:
        pivot[0] = 0
        r4[0] = 0
        r3[0] = 0
        s3[0] = 0
        s4[0] = 0
    
    # Align to daily timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # Trend filter: price vs 200 EMA
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(ema_200[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: reversal at R3 or breakout failure at R4
            if (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or \
               (high[i] >= r4_aligned[i] and close[i] < r4_aligned[i] * 0.995) or \
               close[i] < ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit conditions: reversal at S3 or breakout failure at S4
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or \
               (low[i] <= s4_aligned[i] and close[i] > s4_aligned[i] * 1.005) or \
               close[i] > ema_200[i] or not vol_filter[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: reversal at S3 or breakout above R4 with volume
            if ((low[i] <= s3_aligned[i] and close[i] > s3_aligned[i]) or \
                (high[i] > r4_aligned[i] and close[i] > r4_aligned[i])) and \
               close[i] > ema_200[i] and vol_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: reversal at R3 or breakdown below S4 with volume
            elif ((high[i] >= r3_aligned[i] and close[i] < r3_aligned[i]) or \
                  (low[i] < s4_aligned[i] and close[i] < s4_aligned[i])) and \
                 close[i] < ema_200[i] and vol_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals