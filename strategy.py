#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Daily Pivot Point with Momentum and Volume Confirmation
# Hypothesis: Daily pivot points act as key support/resistance levels where price reacts
# with momentum confirmation (RSI divergence) and volume spikes. Works in both bull and bear
# markets by buying at support with bullish momentum and selling at resistance with bearish momentum.
# Target: 12-25 trades/year (50-100 over 4 years).

name = "6h_daily_pivot_momentum_volume_v1"
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
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    # Calculate daily pivot points (based on previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Standard pivot point calculation
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    r2 = pivot + (daily_high - daily_low)
    s2 = pivot - (daily_high - daily_low)
    r3 = daily_high + 2 * (pivot - daily_low)
    s3 = daily_low - 2 * (daily_high - pivot)
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    pivot = np.roll(pivot, 1)
    r1 = np.roll(r1, 1)
    s1 = np.roll(s1, 1)
    r2 = np.roll(r2, 1)
    s2 = np.roll(s2, 1)
    r3 = np.roll(r3, 1)
    s3 = np.roll(s3, 1)
    
    # Handle first element
    if len(pivot) > 1:
        pivot[0] = pivot[1]
        r1[0] = r1[1]
        s1[0] = s1[1]
        r2[0] = r2[1]
        s2[0] = s2[1]
        r3[0] = r3[1]
        s3[0] = s3[1]
    else:
        pivot[0] = 0
        r1[0] = 0
        s1[0] = 0
        r2[0] = 0
        s2[0] = 0
        r3[0] = 0
        s3[0] = 0
    
    # Align to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
    # Momentum: RSI(14)
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R1 with bearish momentum OR stops above pivot with weak volume
            if (high[i] >= r1_aligned[i] and rsi[i] < 50) or \
               (close[i] < pivot_aligned[i] and not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price reaches S1 with bullish momentum OR stops below pivot with weak volume
            if (low[i] <= s1_aligned[i] and rsi[i] > 50) or \
               (close[i] > pivot_aligned[i] and not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long entry: bounce at S1 with bullish RSI divergence and volume
            if (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i] and
                rsi[i] > 50 and rsi[i] < 70 and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: bounce at R1 with bearish RSI divergence and volume
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i] and
                  rsi[i] < 50 and rsi[i] > 30 and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals