# 6h_200EMA_W10_Pivot_Turn
# Hypothesis: 6-hour price reverses at weekly Pivot S1/R1 when above/below 200-period EMA (long-term trend filter).
# Long when price > 200EMA and crosses above S1 pivot from below.
# Short when price < 200EMA and crosses below R1 pivot from above.
# Exit when price crosses back through the pivot point (PP).
# Uses weekly pivot levels as institutional support/resistance and EMA200 for trend filter.
# Designed for low frequency (target 50-150 trades over 4 years) to minimize fee impact in ranging 2025+ markets.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf  # Note: using align_ltf_to_hlf is incorrect, should be align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 200 EMA for trend filter
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load weekly data ONCE for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly pivot points: PP = (H+L+C)/3, S1 = 2*PP - H, R1 = 2*PP - L
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    pp = (high_1w + low_1w + close_1w) / 3.0
    s1 = 2 * pp - high_1w
    r1 = 2 * pp - low_1w
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical data is NaN
        if (np.isnan(ema200[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        prev_price = close[i-1]
        pp_val = pp_aligned[i]
        s1_val = s1_aligned[i]
        r1_val = r1_aligned[i]
        
        if position == 0:
            # Long: price > EMA200 and crosses above S1 from below
            if price > ema200[i] and prev_price <= s1_val and price > s1_val:
                position = 1
                signals[i] = position_size
            # Short: price < EMA200 and crosses below R1 from above
            elif price < ema200[i] and prev_price >= r1_val and price < r1_val:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below PP (trend failure)
            if prev_price >= pp_val and price < pp_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above PP (trend failure)
            if prev_price <= pp_val and price > pp_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_200EMA_W10_Pivot_Turn"
timeframe = "6h"
leverage = 1.0