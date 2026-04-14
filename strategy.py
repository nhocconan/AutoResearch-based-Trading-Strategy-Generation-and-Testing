#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Reversal with 1d Trend Filter and Volume Spike
# Uses Camarilla pivot levels (H3/L3) from daily timeframe for mean reversion entries
# 1d EMA (50) filters trend direction to avoid counter-trend trades
# Volume spike (>2x 20-period average) confirms institutional interest at pivot levels
# Works in both bull/bear markets: mean reversion in ranges, trend-following in strong trends
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    # Typical Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # where C = (H+L+C)/3 (typical price), but using close as pivot for simplicity
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Pivot point (using close as approximation)
    pivot_1d = close_1d
    # Calculate H3 and L3 levels
    h3_1d = pivot_1d + (high_1d - low_1d) * 1.1 / 4
    l3_1d = pivot_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align H3/L3 to 4h timeframe (wait for daily bar to close)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # 1d EMA (50) for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Volume > 2x average
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price above EMA = uptrend, below = downtrend
        uptrend = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long setup: price at L3 support with volume spike in uptrend
            if price <= l3_1d_aligned[i] * 1.005 and volume_spike[i] and uptrend:
                position = 1
                signals[i] = position_size
            # Short setup: price at H3 resistance with volume spike in downtrend
            elif price >= h3_1d_aligned[i] * 0.995 and volume_spike[i] and not uptrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches midpoint (pivot) or trend changes
            if price >= pivot_1d[i] * 0.995 or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches midpoint (pivot) or trend changes
            if price <= pivot_1d[i] * 1.005 or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_1dEMA_Volume"
timeframe = "4h"
leverage = 1.0