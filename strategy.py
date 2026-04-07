#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla Pivot with 1d Trend and Volume Confirmation
# Hypothesis: Camarilla pivot levels provide high-probability reversal points.
# Combined with 1d EMA50 trend filter to trade in direction of higher timeframe trend.
# Volume confirmation ensures institutional participation at pivot levels.
# Works in both bull and bear markets by only taking trades aligned with 1d trend.
# Targets 15-25 trades/year with disciplined entries to avoid overtrading.

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Previous day's high, low, close for Camarilla calculation
    # We'll calculate daily OHLC from 1d data
    prev_day_high = df_1d['high'].values
    prev_day_low = df_1d['low'].values
    prev_day_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # H2 = close + 0.618 * (high - low)
    # H1 = close + 0.382 * (high - low)
    # L1 = close - 0.382 * (high - low)
    # L2 = close - 0.618 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We focus on H3/L3 for entry and H4/L4 for stop
    
    camarilla_h3 = prev_day_close + 1.1 * (prev_day_high - prev_day_low)
    camarilla_l3 = prev_day_close - 1.1 * (prev_day_high - prev_day_low)
    camarilla_h4 = prev_day_close + 1.5 * (prev_day_high - prev_day_low)
    camarilla_l4 = prev_day_close - 1.5 * (prev_day_high - prev_day_low)
    
    # Align Camarilla levels to 12h timeframe
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for volume SMA
        # Skip if required data not available
        if (np.isnan(ema50_12h[i]) or 
            np.isnan(h3_12h[i]) or 
            np.isnan(l3_12h[i]) or
            np.isnan(h4_12h[i]) or
            np.isnan(l4_12h[i]) or
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1d EMA50 OR touches L4 (stop)
            if close[i] < ema50_12h[i] or close[i] <= l4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 1d EMA50 OR touches H4 (stop)
            if close[i] > ema50_12h[i] or close[i] >= h4_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches L3 with volume confirmation + uptrend
            if (close[i] <= l3_12h[i] and 
                vol_confirm and 
                close[i] > ema50_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches H3 with volume confirmation + downtrend
            elif (close[i] >= h3_12h[i] and 
                  vol_confirm and 
                  close[i] < ema50_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals