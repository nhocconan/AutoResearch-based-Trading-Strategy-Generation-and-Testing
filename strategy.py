#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power with 1w pivot direction filter.
# Uses 1w Camarilla pivot for trend direction (bull/bear/range) to capture major regimes.
# Long when: Alligator bullish (jaw < teeth < lips) AND Elder Bull Power > 0 AND price above 1w pivot point.
# Short when: Alligator bearish (jaw > teeth > lips) AND Elder Bear Power < 0 AND price below 1w pivot point.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 12-25 trades/year.
# Williams Alligator identifies trend via smoothed medians; Elder Ray measures bull/bear power behind moves.
# 1w pivot provides structural regime filter to avoid counter-trend trades in strong markets.
# Works in bull (trend following) and bear (counter-trend at extremes) by aligning with higher timeframe structure.

name = "6h_WilliamsAlligator_ElderRay_1wPivotDir_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for pivot direction
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivot points (using prior week OHLC)
    # We'll calculate pivot from prior week's OHLC for current week's bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Prior week OHLC for current week's pivot
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot_1w = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla-style levels from prior week
    range_1w = prev_high - prev_low
    r1_1w = prev_close + (range_1w * 1.1 / 12)
    s1_1w = prev_close - (range_1w * 1.1 / 12)
    r3_1w = prev_close + (range_1w * 1.1 / 4)
    s3_1w = prev_close - (range_1w * 1.1 / 4)
    
    # Align 1w levels to 6h
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Williams Alligator: SMMA (Smoothed Moving Average) of median price
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMMA) - similar to EMA but with different smoothing
    # Jaw: SMMA(13, 8) - 13-period smoothed, shifted 8 bars
    # Teeth: SMMA(8, 5) - 8-period smoothed, shifted 5 bars
    # Lips: SMMA(5, 3) - 5-period smoothed, shifted 3 bars
    
    def smma(values, period, shift):
        """Calculate Smoothed Moving Average with shift"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        # First value is simple average
        smma_vals = np.full_like(values, np.nan)
        smma_vals[period-1] = np.nanmean(values[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
        for i in range(period, len(values)):
            if not np.isnan(smma_vals[i-1]):
                smma_vals[i] = (smma_vals[i-1] * (period-1) + values[i]) / period
            else:
                smma_vals[i] = np.nan
        # Apply shift
        shifted = np.full_like(smma_vals, np.nan)
        if shift < len(values):
            shifted[shift:] = smma_vals[:-shift]
        return shifted
    
    jaw = smma(median_price, 13, 8)
    teeth = smma(median_price, 8, 5)
    lips = smma(median_price, 5, 3)
    
    # Elder Ray Power
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(pivot_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_pivot = pivot_1w_aligned[i]
        curr_r3 = r3_1w_aligned[i]
        curr_s3 = s3_1w_aligned[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        
        # Williams Alligator conditions
        alligator_bullish = (curr_jaw < curr_teeth) and (curr_teeth < curr_lips)
        alligator_bearish = (curr_jaw > curr_teeth) and (curr_teeth > curr_lips)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish AND Bull Power > 0 AND price above weekly pivot
            if (alligator_bullish and 
                curr_bull_power > 0 and 
                curr_close > curr_pivot):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power < 0 AND price below weekly pivot
            elif (alligator_bearish and 
                  curr_bear_power < 0 and 
                  curr_close < curr_pivot):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Bear Power < 0 OR price breaks below weekly S1
            if (not alligator_bullish or 
                curr_bear_power < 0 or 
                curr_close < s1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Bull Power > 0 OR price breaks above weekly R1
            if (not alligator_bearish or 
                curr_bull_power > 0 or 
                curr_close > r1_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals