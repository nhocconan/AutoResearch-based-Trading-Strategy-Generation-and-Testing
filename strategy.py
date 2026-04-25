#!/usr/bin/env python3
"""
6h_PivotHighLow_Breakout_12hTrend_VolumeConfirm_v1
Hypothesis: Trade breakouts above prior 12h pivot highs and below prior 12h pivot lows on 6h timeframe with 12h EMA50 trend filter and volume confirmation. 
Pivot highs/lows identify significant swing points where price reversed, making them robust support/resistance levels.
In bull markets: buy when price breaks above prior 12h pivot high and price > 12h EMA50. 
In bear markets: sell when price breaks below prior 12h pivot low and price < 12h EMA50. 
Requires volume > 1.3x 20-period average for confirmation to avoid false breakouts.
Exit on opposite pivot level touch or trend reversal. 
Position size: 0.25 to limit drawdown. 
Target: 50-150 total trades over 4 years = 12-37/year.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for pivot points and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need sufficient data for volume average
        return np.zeros(n)
    
    # Calculate 12h EMA50 for HTF trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 20-period average volume for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Calculate pivot highs and lows from 12h data
    # Pivot high: high > previous high AND high > next high
    # Pivot low: low < previous low AND low < next low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Initialize arrays for pivot points
    pivot_high_12h = np.full_like(high_12h, np.nan)
    pivot_low_12h = np.full_like(low_12h, np.nan)
    
    # Calculate pivot points (need at least 3 points: previous, current, next)
    for i in range(1, len(high_12h) - 1):
        # Pivot high: current high is higher than both neighbors
        if high_12h[i] > high_12h[i-1] and high_12h[i] > high_12h[i+1]:
            pivot_high_12h[i] = high_12h[i]
        # Pivot low: current low is lower than both neighbors
        if low_12h[i] < low_12h[i-1] and low_12h[i] < low_12h[i+1]:
            pivot_low_12h[i] = low_12h[i]
    
    # Forward fill pivot points to get the most recent pivot level
    pivot_high_series = pd.Series(pivot_high_12h)
    pivot_low_series = pd.Series(pivot_low_12h)
    pivot_high_ffilled = pivot_high_series.ffill().values
    pivot_low_ffilled = pivot_low_series.ffill().values
    
    # Align pivot levels and indicators to 6h timeframe
    pivot_high_aligned = align_htf_to_ltf(prices, df_12h, pivot_high_ffilled)
    pivot_low_aligned = align_htf_to_ltf(prices, df_12h, pivot_low_ffilled)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50 (50) and volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(pivot_high_aligned[i]) or
            np.isnan(pivot_low_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h HTF trend (bullish = price above 12h EMA50)
        htf_12h_bullish = close[i] > ema_50_12h_aligned[i]
        htf_12h_bearish = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above prior 12h pivot high + 12h uptrend + volume confirmation
            long_setup = (close[i] > pivot_high_aligned[i]) and htf_12h_bullish and volume_confirm
            
            # Short setup: price breaks below prior 12h pivot low + 12h downtrend + volume confirmation
            short_setup = (close[i] < pivot_low_aligned[i]) and htf_12h_bearish and volume_confirm
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price touches prior 12h pivot low (stop) OR 12h trend turns bearish
            if (close[i] <= pivot_low_aligned[i]) or (not htf_12h_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches prior 12h pivot high (stop) OR 12h trend turns bullish
            if (close[i] >= pivot_high_aligned[i]) or (htf_12h_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_PivotHighLow_Breakout_12hTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0