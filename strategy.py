#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
# Long when price breaks above weekly R4 with volume confirmation
# Short when price breaks below weekly S4 with volume confirmation
# Mean reversion fade at R3/S3 when price reaches extreme levels without breakout
# Uses discrete position sizing 0.25 to target ~15-25 trades/year and minimize fee drag
# Weekly Camarilla pivots provide structural support/resistance that works in bull/bear markets
# Volume confirmation filters false breakouts, session filter avoids low-liquidity periods

name = "6h_1w_camarilla_breakout_meanrev_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:  # Need at least 5 weeks for meaningful pivots
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Based on previous week's OHLC
    def rolling_shift(arr, shift):
        return np.roll(arr, shift)
    
    # Previous week's OHLC (shift by 1 to avoid look-ahead)
    prev_high = rolling_shift(high_1w, 1)
    prev_low = rolling_shift(low_1w, 1)
    prev_close = rolling_shift(close_1w, 1)
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance levels
    r3 = pivot + (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    
    # Support levels
    s3 = pivot - (range_hl * 1.1 / 4)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align weekly Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # Volume confirmation: 6h volume > 2x average 6h volume (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > 2.0 * vol_ma_20
    
    # Pre-compute session filter (08-20 UTC) - avoid low liquidity periods
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit long if price falls below weekly R3 (mean reversion) or S4 (stop)
            if close[i] < r3_aligned[i] or close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above weekly S3 (mean reversion) or R4 (stop)
            if close[i] > s3_aligned[i] or close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout entries with volume confirmation
            if close[i] > r4_aligned[i] and volume_confirmed[i]:
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_aligned[i] and volume_confirmed[i]:
                position = -1
                signals[i] = -0.25
            # Mean reversion fade at extremes
            elif close[i] > r3_aligned[i] and close[i] < r4_aligned[i]:
                # Fade at R3 resistance if not breaking out
                position = -1
                signals[i] = -0.25
            elif close[i] < s3_aligned[i] and close[i] > s4_aligned[i]:
                # Fade at S3 support if not breaking down
                position = 1
                signals[i] = 0.25
    
    return signals