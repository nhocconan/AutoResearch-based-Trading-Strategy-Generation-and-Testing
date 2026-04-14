#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily Pivot Reversion with Weekly Trend Filter
# Uses daily pivot points (S1/R1) as mean reversion levels - price tends to revert to these levels
# Weekly EMA (21) provides trend filter to avoid counter-trend trades
# Pivot-based approach works in both bull/bear markets by capturing reversion to key levels
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load daily data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points (standard formula)
    # Pivot = (High + Low + Close) / 3
    # S1 = 2*Pivot - High
    # R1 = 2*Pivot - Low
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    s1 = 2 * pivot - df_1d['high']
    r1 = 2 * pivot - df_1d['low']
    
    # Align pivot levels to 1d timeframe (already daily, so just forward fill for intraday)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA (21) for trend direction
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need at least 1 day of data)
    start = 1
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade in direction of weekly EMA
        above_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: price touches or goes below S1 with uptrend filter
            if price <= s1_aligned[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price touches or goes above R1 with downtrend filter
            elif price >= r1_aligned[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches pivot level (mean reversion complete) or trend changes
            if price >= pivot_aligned[i] or price < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches pivot level (mean reversion complete) or trend changes
            if price <= pivot_aligned[i] or price > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Pivot_Reversion_WeeklyEMA_Trend"
timeframe = "1d"
leverage = 1.0