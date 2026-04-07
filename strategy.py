#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Camarilla pivot reversal with 1d volume spike and 12h trend filter
# Hypothesis: Camarilla levels act as strong support/resistance; reversals confirmed by volume and higher timeframe trend work in both bull (buying dips) and bear (selling rallies).
# Target: 25-40 trades/year to minimize fee drag.
name = "4h_camarilla_pivot_1d_volume_12h_trend_v1"
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
    
    # Get daily data for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: Range = prev_day_high - prev_day_low
    # L4 = close + 1.1 * Range * 1.1/2
    # L3 = close + 1.1 * Range * 1.1/4
    # L2 = close + 1.1 * Range * 1.1/6
    # L1 = close + 1.1 * Range * 1.1/12
    # H1 = close - 1.1 * Range * 1.1/12
    # H2 = close - 1.1 * Range * 1.1/6
    # H3 = close - 1.1 * Range * 1.1/4
    # H4 = close - 1.1 * Range * 1.1/2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Calculate all 8 levels
    H4 = prev_close - 1.1 * prev_range * 1.1 / 2
    H3 = prev_close - 1.1 * prev_range * 1.1 / 4
    H2 = prev_close - 1.1 * prev_range * 1.1 / 6
    H1 = prev_close - 1.1 * prev_range * 1.1 / 12
    L1 = prev_close + 1.1 * prev_range * 1.1 / 12
    L2 = prev_close + 1.1 * prev_range * 1.1 / 6
    L3 = prev_close + 1.1 * prev_range * 1.1 / 4
    L4 = prev_close + 1.1 * prev_range * 1.1 / 2
    
    # Align Camarilla levels to 4h
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Daily volume spike: current day volume > 1.5 * 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (vol_ma_1d * 1.5)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 12h trend filter: EMA(20) slope
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20).mean().values
    ema_slope_12h = np.diff(ema_12h, prepend=ema_12h[0])
    ema_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(ema_slope_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches H3 (take profit) or breaks below L1 (stop)
            if close[i] >= H3_aligned[i] or close[i] <= L1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price reaches L3 (take profit) or breaks above H1 (stop)
            if close[i] <= L3_aligned[i] or close[i] >= H1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price touches L4 + volume spike + 12h uptrend
            if (close[i] <= L4_aligned[i] and 
                vol_spike_1d_aligned[i] and 
                ema_slope_12h_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Enter short: price touches H4 + volume spike + 12h downtrend
            elif (close[i] >= H4_aligned[i] and 
                  vol_spike_1d_aligned[i] and 
                  ema_slope_12h_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals