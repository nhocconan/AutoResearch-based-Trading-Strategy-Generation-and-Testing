#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Camarilla Pivot + 1d Trend + Volume Confirmation
# Hypothesis: Camarilla pivot levels on 6-hour timeframe provide reversal points in ranging markets
# while acting as breakout levels in trending markets. Combined with 1-day EMA trend filter and
# volume confirmation, this strategy adapts to both bull and bear markets by fading at inner levels
# (R3/S3) and breaking out at outer levels (R4/S4). 6h timeframe reduces noise while maintaining
# responsiveness. Target: 12-37 trades/year (50-150 over 4 years) to minimize fee drag.
name = "6h_camarilla_pivot_1d_trend_volume_v1"
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
    
    # Get 1-day data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 6h timeframe using previous day's OHLC
    # We need to align previous day's data to each 6h bar
    # For each 6h bar, use previous day's OHLC to calculate today's Camarilla levels
    prev_day_open = df_1d['open'].shift(1).values  # Previous day's open
    prev_day_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_day_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate pivot point and ranges
    pivot_point = (prev_day_high + prev_day_low + prev_day_close) / 3
    range_hl = prev_day_high - prev_day_low
    
    # Camarilla levels
    r4 = pivot_point + (range_hl * 1.1 / 2)
    r3 = pivot_point + (range_hl * 1.1 / 4)
    s3 = pivot_point - (range_hl * 1.1 / 4)
    s4 = pivot_point - (range_hl * 1.1 / 2)
    
    # Align daily levels to 6h timeframe (already shifted by 1 for previous day's data)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1-day EMA(50) for trend filter
    daily_close = df_1d['close'].values
    daily_ema = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    daily_ema_6h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(daily_ema_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches S3 (take profit) or breaks below S4 (stop) or trend turns bearish
            if close[i] <= s3_6h[i] or close[i] < s4_6h[i] or close[i] < daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches R3 (take profit) or breaks above R4 (stop) or trend turns bullish
            if close[i] >= r3_6h[i] or close[i] > r4_6h[i] or close[i] > daily_ema_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Fade at inner levels (R3/S3) in ranging markets
                # Long: price crosses above S3 with rejection of S4
                if close[i] > s3_6h[i] and close[i] < s4_6h[i]:
                    # Additional confirmation: price should be below pivot in ranging market
                    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
                    if not np.isnan(pivot_point_6h[i]) and close[i] < pivot_point_6h[i]:
                        position = 1
                        signals[i] = 0.25
                # Short: price crosses below R3 with rejection of R4
                elif close[i] < r3_6h[i] and close[i] > r4_6h[i]:
                    # Additional confirmation: price should be above pivot in ranging market
                    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
                    if not np.isnan(pivot_point_6h[i]) and close[i] > pivot_point_6h[i]:
                        position = -1
                        signals[i] = -0.25
                # Breakout at outer levels (R4/S4) in trending markets
                # Long: price breaks above R4 with trend confirmation
                elif close[i] > r4_6h[i] and close[i] > daily_ema_6h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below S4 with trend confirmation
                elif close[i] < s4_6h[i] and close[i] < daily_ema_6h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals