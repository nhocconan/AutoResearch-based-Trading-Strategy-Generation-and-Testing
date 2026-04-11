#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: Daily Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (L3/L4 for long, H3/H4 for short) act as support/resistance.
# Weekly trend filter (price above/below weekly SMA50) ensures trades align with higher timeframe trend.
# Volume confirmation (daily volume > 1.5x 20-day average) filters low-conviction moves.
# Designed for low trade frequency (~10-25/year) to minimize fee drift and work in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly SMA50 for trend filter
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = high
    low_1d = low
    close_1d = close
    
    # Calculate Camarilla levels for each day using previous day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.0 * (high - low)
    # L3 = close - 1.0 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # We use shifted values to avoid look-ahead (yesterday's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # avoid NaN on first bar
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.0 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.0 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Daily volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if any required data is invalid
        if np.isnan(sma_50_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or \
           np.isnan(vol_avg_20[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter: price above/below weekly SMA50
        weekly_uptrend = close_1d[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close_1d[i] < sma_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Entry conditions
        # Long: price crosses above Camarilla L3 in uptrend with volume
        if weekly_uptrend and vol_confirm and close_1d[i] > camarilla_l3[i] and close_1d[i-1] <= camarilla_l3[i-1] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price crosses below Camarilla H3 in downtrend with volume
        elif weekly_downtrend and vol_confirm and close_1d[i] < camarilla_h3[i] and close_1d[i-1] >= camarilla_h3[i-1] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite Camarilla level touch (mean reversion)
        elif position == 1 and close_1d[i] < camarilla_l4[i]:  # reversed to L4
            position = 0
            signals[i] = 0.0
        elif position == -1 and close_1d[i] > camarilla_h4[i]:  # reversed to H4
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals