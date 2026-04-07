#!/usr/bin/env python3
"""
6h_weekly_pivot_rsi_divergence_v1
Hypothesis: On 6h timeframe, use weekly pivot levels as support/resistance and RSI divergence for entry signals.
- In uptrend (price > weekly SMA50): look for bullish RSI divergence at weekly support levels (S1/S2) for long entries
- In downtrend (price < weekly SMA50): look for bearish RSI divergence at weekly resistance levels (R1/R2) for short entries
- Weekly pivot levels calculated from previous week's OHLC to avoid look-ahead
- RSI divergence: price makes new low/high but RSI makes higher low/lower high (bullish/bearish divergence)
- Volume confirmation: require volume > 1.5x 24-period average to confirm genuine tests
- Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25
- Designed to work in both bull and bear markets by trading mean reversion at key weekly levels with momentum confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_pivot_rsi_divergence_v1"
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
    
    # Weekly data for pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly SMA50 for trend filter
    sma_50 = df_weekly['close'].rolling(window=50, min_periods=50).mean()
    
    # Align weekly SMA50 to 6h timeframe
    sma_50_aligned = align_htf_to_ltf(prices, df_weekly, sma_50.values)
    
    # Calculate weekly pivot levels from previous week
    # Using previous week's OHLC to avoid look-ahead
    prev_weekly_close = df_weekly['close'].shift(1)
    prev_weekly_high = df_weekly['high'].shift(1)
    prev_weekly_low = df_weekly['low'].shift(1)
    
    # Pivot point
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    range_val = prev_weekly_high - prev_weekly_low
    
    # Weekly pivot levels (using standard pivot formula)
    # Resistance levels
    r1 = 2 * pivot - prev_weekly_low
    r2 = pivot + (prev_weekly_high - prev_weekly_low)
    r3 = prev_weekly_high + 2 * (pivot - prev_weekly_low)
    # Support levels
    s1 = 2 * pivot - prev_weekly_high
    s2 = pivot - (prev_weekly_high - prev_weekly_low)
    s3 = prev_weekly_low - 2 * (prev_weekly_high - pivot)
    
    # Align all levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot.values)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3.values)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3.values)
    
    # RSI calculation (14-period)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation (24-period average on 6h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(sma_50_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(rsi_values[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S2 or trend turns bearish
            if close[i] < s2_aligned[i] or close[i] < sma_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R2 or trend turns bullish
            if close[i] > r2_aligned[i] or close[i] > sma_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Check for RSI divergence (need at least 3 periods to detect)
            if i >= 3:
                # Bullish divergence: price makes lower low, RSI makes higher low
                bullish_div = (low[i] < low[i-1] and low[i-1] < low[i-2] and 
                              rsi_values[i] > rsi_values[i-1] and rsi_values[i-1] > rsi_values[i-2])
                # Bearish divergence: price makes higher high, RSI makes lower high
                bearish_div = (high[i] > high[i-1] and high[i-1] > high[i-2] and 
                              rsi_values[i] < rsi_values[i-1] and rsi_values[i-1] < rsi_values[i-2])
            else:
                bullish_div = False
                bearish_div = False
            
            # Long entry: bullish RSI divergence at weekly support in uptrend
            if (bullish_div and 
                (abs(low[i] - s1_aligned[i]) < 0.005 * s1_aligned[i] or abs(low[i] - s2_aligned[i]) < 0.005 * s2_aligned[i]) and  # near S1/S2
                vol_confirm and 
                close[i] > sma_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: bearish RSI divergence at weekly resistance in downtrend
            elif (bearish_div and 
                  (abs(high[i] - r1_aligned[i]) < 0.005 * r1_aligned[i] or abs(high[i] - r2_aligned[i]) < 0.005 * r2_aligned[i]) and  # near R1/R2
                  vol_confirm and 
                  close[i] < sma_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals