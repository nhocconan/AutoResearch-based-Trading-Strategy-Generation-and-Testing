#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Weekly Camarilla Pivot with Daily Trend Filter
# Hypothesis: Weekly Camarilla levels (R3/S3, R4/S4) act as strong support/resistance.
# Breakouts above R4 with daily trend confirmation (price > 20 EMA) indicate bullish continuation.
# Breakdowns below S4 with daily trend confirmation (price < 20 EMA) indicate bearish continuation.
# Uses weekly pivot levels for structure and daily EMA for trend filter to avoid counter-trend trades.
# Target: 15-30 trades/year (60-120 over 4 years) to stay within trade limits.

name = "6h_weekly_camarilla_pivot_daily_trend_v1"
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
    
    # Get weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Calculate weekly data (previous week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Shift by 1 to use previous week's data (avoid look-ahead)
    prev_weekly_high = np.roll(weekly_high, 1)
    prev_weekly_low = np.roll(weekly_low, 1)
    prev_weekly_close = np.roll(weekly_close, 1)
    prev_weekly_high[0] = prev_weekly_high[1] if len(prev_weekly_high) > 1 else 0
    prev_weekly_low[0] = prev_weekly_low[1] if len(prev_weekly_low) > 1 else 0
    prev_weekly_close[0] = prev_weekly_close[1] if len(prev_weekly_close) > 1 else 0
    
    # Calculate weekly Camarilla pivot levels
    weekly_range = prev_weekly_high - prev_weekly_low
    weekly_close_prev = prev_weekly_close
    weekly_r3 = weekly_close_prev + (weekly_range * 1.1 / 2)
    weekly_s3 = weekly_close_prev - (weekly_range * 1.1 / 2)
    weekly_r4 = weekly_close_prev + (weekly_range * 1.1)
    weekly_s4 = weekly_close_prev - (weekly_range * 1.1)
    
    # Align to 6h timeframe (use previous week's levels)
    weekly_r4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r4)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s4)
    
    # Daily trend filter: price above/below 20 EMA
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or 
            np.isnan(ema_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below S3 or trend turns bearish
            if (close[i] < weekly_s3_aligned[i] or close[i] < ema_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above R3 or trend turns bullish
            if (close[i] > weekly_r3_aligned[i] or close[i] > ema_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above R4 with bullish trend
            if ((high[i] > weekly_r4_aligned[i] or close[i] > weekly_r4_aligned[i]) and 
                close[i] > ema_20[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with bearish trend
            elif ((low[i] < weekly_s4_aligned[i] or close[i] < weekly_s4_aligned[i]) and 
                  close[i] < ema_20[i]):
                position = -1
                signals[i] = -0.25
    
    return signals