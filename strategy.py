#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power + 12h Trend Filter
# - Primary: 6h timeframe for balance of signal frequency and fee drag
# - HTF: 12h for trend direction (EMA50 slope) to avoid counter-trend trades
# - Long: 6h Bull Power > 0 AND 6h Bear Power < 0 AND 12h EMA50 rising
# - Short: 6h Bear Power < 0 AND 6h Bull Power < 0 AND 12h EMA50 falling
# - Exit: Opposite Elder Ray signal or 12h EMA50 flattening (slope near zero)
# - Position sizing: 0.25 (discrete level)
# - Target: 80-180 total trades over 4 years (20-45/year) - within 6h sweet spot
# - Works in bull/bear: Elder Ray captures momentum shifts; 12h EMA filter avoids whipsaws in ranging markets

name = "6h_12h_elderray_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 6h OHLCV
    open_6h = prices['open'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    volume_6h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 6h Elder Ray Power (13-period EMA)
    close_6h_series = pd.Series(close_6h)
    ema13_6h = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_6h = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power_6h = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 12h EMA50 slope (5-period change) for trend direction
    ema50_slope = np.zeros_like(ema50_12h)
    ema50_slope[5:] = ema50_12h[5:] - ema50_12h[:-5]  # 5-bar slope
    
    # Align HTF data to 6h timeframe
    bull_power_6h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_6h)
    bear_power_6h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_6h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema50_slope)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h_aligned[i]) or np.isnan(bear_power_6h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema50_slope_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter conditions
        # Rising trend: EMA50 slope > 0.001 * price (avoid noise)
        rising_trend = ema50_slope_aligned[i] > 0.001 * close_6h[i]
        # Falling trend: EMA50 slope < -0.001 * price
        falling_trend = ema50_slope_aligned[i] < -0.001 * close_6h[i]
        # Flat trend: |slope| <= 0.001 * price
        flat_trend = np.abs(ema50_slope_aligned[i]) <= 0.001 * close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power positive AND Bear Power negative AND rising 12h trend
            if (bull_power_6h_aligned[i] > 0 and bear_power_6h_aligned[i] < 0 and rising_trend):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power negative AND Bull Power negative AND falling 12h trend
            elif (bear_power_6h_aligned[i] < 0 and bull_power_6h_aligned[i] < 0 and falling_trend):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Opposite Elder Ray signal (momentum exhaustion)
            # 2. 12h trend flattens (loses directional momentum)
            
            if position == 1:  # Long position
                exit_condition = (
                    bull_power_6h_aligned[i] <= 0 or  # Bull Power exhausted
                    bear_power_6h_aligned[i] >= 0 or  # Bear Power turned positive
                    flat_trend                        # 12h trend flattened
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                exit_condition = (
                    bear_power_6h_aligned[i] >= 0 or  # Bear Power exhausted
                    bull_power_6h_aligned[i] >= 0 or  # Bull Power turned positive
                    flat_trend                        # 12h trend flattened
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals