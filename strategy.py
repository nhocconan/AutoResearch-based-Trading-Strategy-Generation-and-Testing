#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: 12-hour Camarilla pivot levels with 1-day trend filter and volume confirmation.
Uses daily EMA(50) for trend direction, Camarilla pivot levels (based on previous day's high/low/close)
for support/resistance, and volume > 1.5x average for confirmation.
In bull markets: longs near L3/L4 with daily uptrend.
In bear markets: shorts near H3/H4 with daily downtrend.
Camarilla levels work well in ranging markets and provide clear reversal points.
Target: 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

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
    
    # Calculate Camarilla pivot levels from previous day
    # Need daily high/low/close for previous day
    # For each 12h bar, we need the previous day's OHLC
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (for today's Camarilla levels)
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Calculate Camarilla levels
    # Range = previous day high - previous day low
    rng = prev_high - prev_low
    
    # Camarilla levels
    # H4 = close + 1.1 * range * 1.1/2
    # H3 = close + 1.1 * range * 1.1/4
    # H2 = close + 1.1 * range * 1.1/6
    # H1 = close + 1.1 * range * 1.1/12
    # L1 = close - 1.1 * range * 1.1/12
    # L2 = close - 1.1 * range * 1.1/6
    # L3 = close - 1.1 * range * 1.1/4
    # L4 = close - 1.1 * range * 1.1/2
    
    # Actually, standard Camarilla:
    # H4 = close + (high - low) * 1.1/2
    # H3 = close + (high - low) * 1.1/4
    # H2 = close + (high - low) * 1.1/6
    # H1 = close + (high - low) * 1.1/12
    # L1 = close - (high - low) * 1.1/12
    # L2 = close - (high - low) * 1.1/6
    # L3 = close - (high - low) * 1.1/4
    # L4 = close - (high - low) * 1.1/2
    
    H4 = prev_close + rng * 1.1 / 2
    H3 = prev_close + rng * 1.1 / 4
    H2 = prev_close + rng * 1.1 / 6
    H1 = prev_close + rng * 1.1 / 12
    L1 = prev_close - rng * 1.1 / 12
    L2 = prev_close - rng * 1.1 / 6
    L3 = prev_close - rng * 1.1 / 4
    L4 = prev_close - rng * 1.1 / 2
    
    # Align daily Camarilla levels to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Daily trend filter: EMA(50)
    daily_ema = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    daily_ema_12h = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from index 1 to ensure we have previous day data
    for i in range(1, n):
        # Skip if required data not available
        if (np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(daily_ema_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter
        above_daily_ema = close[i] > daily_ema_12h[i]
        below_daily_ema = close[i] < daily_ema_12h[i]
        
        if position == 1:  # Long position
            # Exit: price below L3 or loss of daily uptrend
            if close[i] < L3_12h[i] or not above_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price above H3 or loss of daily downtrend
            if close[i] > H3_12h[i] or not below_daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price crosses above L3 with volume and daily uptrend
            if (close[i] > L3_12h[i] and 
                close[i-1] <= L3_12h[i-1] and  # crossed above L3
                vol_confirm and 
                above_daily_ema):
                position = 1
                signals[i] = 0.25
            # Short entry: price crosses below H3 with volume and daily downtrend
            elif (close[i] < H3_12h[i] and 
                  close[i-1] >= H3_12h[i-1] and  # crossed below H3
                  vol_confirm and 
                  below_daily_ema):
                position = -1
                signals[i] = -0.25
    
    return signals