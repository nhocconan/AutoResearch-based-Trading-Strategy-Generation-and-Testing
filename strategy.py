#!/usr/bin/env python3
"""
1d Weekly Pivot + Volume Confirmation + ATR Stop
Hypothesis: Weekly pivot levels (from previous week) act as strong support/resistance.
In bull markets: buy breakouts above weekly R4 with volume.
In bear markets: sell breakdowns below weekly S4 with volume.
In ranging markets: fade touches of weekly R3/S3 with volume confirmation.
Uses 1d trend filter (EMA50) to align with higher timeframe bias.
Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weeklypivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Weekly pivot levels (based on previous week's OHLC)
    # We need daily data to calculate weekly pivot, but we'll use 1d data
    # Since we're on 1d timeframe, we can calculate weekly pivot directly
    # by grouping into weeks, but to avoid look-ahead, we use previous week's data
    
    # Calculate weekly pivot using previous week's Monday-Friday data
    # We'll track weekly high, low, close as we go
    weekly_high = np.full(n, np.nan)
    weekly_low = np.full(n, np.nan)
    weekly_close = np.full(n, np.nan)
    
    # Week tracking variables
    week_start_idx = 0
    current_week_high = -np.inf
    current_week_low = np.inf
    current_week_close = 0
    
    for i in range(n):
        # Update current week's high/low
        if high[i] > current_week_high:
            current_week_high = high[i]
        if low[i] < current_week_low:
            current_week_low = low[i]
        current_week_close = close[i]
        
        # Check if we've reached end of week (Friday)
        # Assuming 5 trading days per week
        if i >= 4 and (i - week_start_idx) >= 4:  # 5 days completed
            # Store previous week's data for current bar
            weekly_high[i] = current_week_high
            weekly_low[i] = current_week_low
            weekly_close[i] = current_week_close
            
            # Reset for next week
            week_start_idx = i + 1
            current_week_high = -np.inf
            current_week_low = np.inf
            current_week_close = 0
        elif i < 5:  # First 4 days, not enough for weekly pivot yet
            weekly_high[i] = np.nan
            weekly_low[i] = np.nan
            weekly_close[i] = np.nan
    
    # Calculate weekly pivot levels from previous week's data
    # Pivot = (H + L + C)/3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = 3*P - 2*L, S4 = 3*H - 2*L
    
    pivot = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    s2 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    r4 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(5, n):  # Start from where we have weekly data
        if not (np.isnan(weekly_high[i]) or np.isnan(weekly_low[i]) or np.isnan(weekly_close[i])):
            wh = weekly_high[i]
            wl = weekly_low[i]
            wc = weekly_close[i]
            
            p = (wh + wl + wc) / 3.0
            pivot[i] = p
            r1[i] = 2*p - wl
            s1[i] = 2*p - wh
            r2[i] = p + (wh - wl)
            s2[i] = p - (wh - wl)
            r3[i] = wh + 2*(p - wl)
            s3[i] = wl - 2*(wh - p)
            r4[i] = 3*p - 2*wl
            s4[i] = 3*wh - 2*wl
    
    # 1d EMA50 for trend bias
    ema = np.full(n, np.nan)
    if n >= 50:
        ema[49] = np.mean(close[:50])
        for i in range(50, n):
            ema[i] = (close[i] * 2 + ema[i-1] * 18) / 20
    
    # Trend bias: above EMA = bullish, below = bearish
    trend_bias = np.where(close > ema, 1, -1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # Need enough data for weekly pivot
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(pivot[i]) or 
            np.isnan(r4[i]) or np.isnan(s4[i]) or
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(trend_bias[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price drops below S3 (mean reversion) OR against trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < s3[i] or
                trend_bias[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above R3 (mean reversion) OR against trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > r3[i] or
                trend_bias[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 10 bars flat
            if bars_since_entry >= 10:
                # Breakout entries: R4/S4 with trend
                bull_breakout = close[i] > r4[i]
                bear_breakout = close[i] < s4[i]
                
                # Mean reversion entries: R3/S3 counter-trend (fade)
                # Only in ranging markets - we'll use proximity to pivot as proxy
                pivot_range = r1[i] - s1[i]
                near_pivot = abs(close[i] - pivot[i]) < pivot_range * 0.3
                
                # Long: breakout with trend OR mean reversion at S3 with volume
                if (bull_breakout and trend_bias[i] == 1 and volume_filter) or \
                   (close[i] > s3[i] and close[i] < pivot[i] and 
                    near_pivot and volume_filter and trend_bias[i] == -1):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: breakdown with trend OR mean reversion at R3 with volume
                elif (bear_breakout and trend_bias[i] == -1 and volume_filter) or \
                     (close[i] < r3[i] and close[i] > pivot[i] and 
                      near_pivot and volume_filter and trend_bias[i] == 1):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals