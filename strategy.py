#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Strategy: 1d Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong support/resistance. Long at L3/S1, short at H3/S4.
# Weekly trend filter ensures trades align with higher timeframe momentum. Volume confirmation
# filters out low-conviction moves. Designed for low trade frequency (~10-20/year) to minimize
# fee drag and survive bear markets via selective short entries.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily Camarilla pivot levels from previous day
    # Pivot levels based on previous day's OHLC
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    # L3 = pivot - (range * 1.1 / 4)
    # H3 = pivot + (range * 1.1 / 4)
    # L4 = pivot - (range * 1.1 / 2)
    # H4 = pivot + (range * 1.1 / 2)
    L3 = pivot - (range_val * 1.1 / 4)
    H3 = pivot + (range_val * 1.1 / 4)
    L4 = pivot - (range_val * 1.1 / 2)
    H4 = pivot + (range_val * 1.1 / 2)
    
    # Align weekly trend to daily
    weekly_uptrend = ema_20_1w_aligned > np.roll(ema_20_1w_aligned, 1)
    weekly_uptrend[0] = False
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to have previous day data
        # Skip if any required data is invalid
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(L3[i]) or np.isnan(H3[i]) or \
           np.isnan(L4[i]) or np.isnan(H4[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        if i >= 20:
            vol_avg_20 = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > 1.5 * vol_avg_20
        else:
            vol_confirm = False
        
        # Entry conditions
        # Long: Price <= L3 (strong support) AND weekly uptrend AND volume confirmation
        if close[i] <= L3[i] and weekly_uptrend[i] and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price >= H3 (strong resistance) AND weekly downtrend AND volume confirmation
        elif close[i] >= H3[i] and not weekly_uptrend[i] and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price reaches opposite H4/L4 level (failed breakout/reversal)
        elif position == 1 and close[i] >= H4[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= L4[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals