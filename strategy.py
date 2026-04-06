#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal + 12h trend filter + volume confirmation
# Uses daily Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts) from 12h timeframe.
# Trend filter: 12h EMA(20) ensures alignment with medium-term trend.
# Volume confirmation: current volume > 1.3x 20-period average filters low-quality signals.
# Works in bull markets via R4 breakouts and in bear markets via S4 breakdowns.
# Also captures reversals at R3/S3 during ranging markets.
# Target: 80-180 trades over 4 years (20-45/year).

name = "6h_camarilla_pivot_12h_trend_vol_v1"
timeframe = "6h"
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
    
    # Get 12h data for Camarilla pivots and trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for each 12h bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = Pivot + 1.1 * Range / 2
    # S3 = Pivot - 1.1 * Range / 2
    # R4 = Pivot + 1.1 * Range
    # S4 = Pivot - 1.1 * Range
    pivot_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    r3_12h = pivot_12h + 1.1 * range_12h / 2
    s3_12h = pivot_12h - 1.1 * range_12h / 2
    r4_12h = pivot_12h + 1.1 * range_12h
    s4_12h = pivot_12h - 1.1 * range_12h
    
    # Trend filter: 12h EMA(20)
    ema_20_12h = np.full(len(close_12h), np.nan)
    for i in range(len(close_12h)):
        if i < 19:
            ema_20_12h[i] = np.nan
        elif i == 19:
            ema_20_12h[i] = np.mean(close_12h[0:20])
        else:
            ema_20_12h[i] = close_12h[i] * 2/(20+1) + ema_20_12h[i-1] * (1 - 2/(20+1))
    
    # Align 12h data to 6h timeframe
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if 12h data not available
        if (np.isnan(pivot_12h_aligned[i]) or np.isnan(r3_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below S3 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < s3_12h_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R3 or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > r3_12h_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Breakout above R4 with uptrend
                if (close[i] > r4_12h_aligned[i] and close[i-1] <= r4_12h_aligned[i] and 
                    close[i] > ema_20_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Breakdown below S4 with downtrend
                elif (close[i] < s4_12h_aligned[i] and close[i-1] >= s4_12h_aligned[i] and 
                      close[i] < ema_20_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Reversal at R3 in downtrend
                elif (close[i] < r3_12h_aligned[i] and close[i-1] >= r3_12h_aligned[i] and 
                      close[i] < ema_20_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                # Reversal at S3 in uptrend
                elif (close[i] > s3_12h_aligned[i] and close[i-1] <= s3_12h_aligned[i] and 
                      close[i] > ema_20_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
    
    return signals