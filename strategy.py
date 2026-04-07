#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Daily Pivot Breakout with Volume and Choppiness Filter
# Hypothesis: Daily pivot levels act as strong support/resistance. Price breaking above R1 or below S1 with high volume indicates institutional participation, leading to continuation. Choppiness filter avoids whipsaws in ranging markets. Works in bull/bear: in bull, breaks above R1 continue up; in bear, breaks below S1 continue down. Volume confirms real breakouts; chop filter avoids false signals.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_daily_pivot_breakout_volume_chop_v2"
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
    
    # Get daily data for pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily data (previous day's OHLC)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high[0] = prev_daily_high[1] if len(prev_daily_high) > 1 else 0
    prev_daily_low[0] = prev_daily_low[1] if len(prev_daily_low) > 1 else 0
    prev_daily_close[0] = prev_daily_close[1] if len(prev_daily_close) > 1 else 0
    
    # Calculate daily pivot points
    daily_pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    daily_r1 = (2 * daily_pivot) - prev_daily_low
    daily_s1 = (2 * daily_pivot) - prev_daily_high
    
    # Align to 4h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_daily, daily_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, daily_r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, daily_s1)
    
    # Volume filter: volume > 1.8x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.8 * vol_ma)
    
    # Choppiness filter: avoid ranging markets
    hl_range = np.maximum(high, low) - np.minimum(high, low)
    atr = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    hl_prev_close = np.roll(close, 1)
    hl_prev_close[0] = close[0]
    true_range = np.maximum(high - low, np.maximum(np.abs(high - hl_prev_close), np.abs(low - hl_prev_close)))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    chop = 100 * np.log14(sum(pd.Series(true_range).rolling(14).sum()) / (atr * 14)) / np.log14(14)
    chop_series = pd.Series(chop)
    chop_values = chop_series.fillna(50).values
    chop_filter = chop_values < 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to S1 or volatility drops or chop increases
            if close[i] <= s1_aligned[i] or not vol_filter[i] or chop_values[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to R1 or volatility drops or chop increases
            if close[i] >= r1_aligned[i] or not vol_filter[i] or chop_values[i] >= 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R1 with volume in trending market
            if high[i] > r1_aligned[i] and close[i] > r1_aligned[i] and vol_filter[i] and chop_values[i] < 61.8:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S1 with volume in trending market
            elif low[i] < s1_aligned[i] and close[i] < s1_aligned[i] and vol_filter[i] and chop_values[i] < 61.8:
                position = -1
                signals[i] = -0.25
    
    return signals