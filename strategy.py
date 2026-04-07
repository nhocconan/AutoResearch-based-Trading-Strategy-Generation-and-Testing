#!/usr/bin/env python3
"""
1d_camellia_pivot_1w_trend_volume_v1
Hypothesis: On 1d timeframe, enter long at Camarilla L3 support with above-average volume and weekly uptrend, enter short at H3 resistance with above-average volume and weekly downtrend. Exit at L4/H4 levels. Uses weekly EMA filter to avoid counter-trend trades. Designed for 10-25 trades/year to minimize fee drift while capturing mean reversion in ranging markets and trend continuation in strong moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camellia_pivot_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla levels (using previous day's range)
    # Camarilla: H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # H2 = close + 0.55*(high-low), L2 = close - 0.55*(high-low)
    # H1 = close + 0.275*(high-low), L1 = close - 0.275*(high-low)
    # Pivot = (high + low + close) / 3
    
    # Use previous day's data to calculate today's levels (no look-ahead)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    daily_range = prev_high - prev_low
    H4 = prev_close + 1.5 * daily_range
    L4 = prev_close - 1.5 * daily_range
    H3 = prev_close + 1.1 * daily_range
    L3 = prev_close - 1.1 * daily_range
    H2 = prev_close + 0.55 * daily_range
    L2 = prev_close - 0.55 * daily_range
    H1 = prev_close + 0.275 * daily_range
    L1 = prev_close - 0.275 * daily_range
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or np.isnan(H4[i]) or np.isnan(L4[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(weekly_ema_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H4 (take profit) or L4 (stop loss)
            if close[i] >= H4[i] or close[i] <= L4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches L4 (take profit) or H4 (stop loss)
            if close[i] <= L4[i] or close[i] >= H4[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price at L3 support with weekly uptrend
                if close[i] <= L3[i] and weekly_ema_aligned[i] > weekly_ema_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short: price at H3 resistance with weekly downtrend
                elif close[i] >= H3[i] and weekly_ema_aligned[i] < weekly_ema_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals