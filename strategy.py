# 6h Monthly Pivot + Volume Spike + Trend Filter
# Hypothesis: Monthly pivot levels (from monthly chart) act as strong structural support/resistance.
# In trending markets, price pulls back to these levels for continuation entries.
# In ranging markets, reversals occur at these levels.
# Volume spikes confirm institutional interest at key levels.
# Trend filter (1w EMA) ensures alignment with higher timeframe momentum.
# Timeframe: 6h balances trade frequency (~15-35/year) with signal quality.
# Works in bull/bear: uses price action at key levels rather than trend direction.

name = "6h_MonthlyPivot_Volume_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === MONTHLY DATA FOR PIVOT LEVELS ===
    df_1M = get_htf_data(prices, '1M')
    high_1M = df_1M['high'].values
    low_1M = df_1M['low'].values
    close_1M = df_1M['close'].values
    
    # Monthly pivot points (standard calculation)
    pivot_1M = (high_1M + low_1M + close_1M) / 3.0
    r1_1M = 2 * pivot_1M - low_1M
    s1_1M = 2 * pivot_1M - high_1M
    r2_1M = pivot_1M + (high_1M - low_1M)
    s2_1M = pivot_1M - (high_1M - low_1M)
    
    # Align monthly levels to 6h timeframe
    pivot_1M_aligned = align_htf_to_ltf(prices, df_1M, pivot_1M)
    r1_1M_aligned = align_htf_to_ltf(prices, df_1M, r1_1M)
    s1_1M_aligned = align_htf_to_ltf(prices, df_1M, s1_1M)
    r2_1M_aligned = align_htf_to_ltf(prices, df_1M, r2_1M)
    s2_1M_aligned = align_htf_to_ltf(prices, df_1M, s2_1M)
    
    # === WEEKLY TREND FILTER (EMA 21) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # For volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot_1M_aligned[i]) or np.isnan(r1_1M_aligned[i]) or 
            np.isnan(s1_1M_aligned[i]) or np.isnan(r2_1M_aligned[i]) or 
            np.isnan(s2_1M_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price bounces off S1 or S2 with volume and above weekly EMA
            if ((close[i] > s1_1M_aligned[i] and low[i] <= s1_1M_aligned[i] * 1.005) or
                (close[i] > s2_1M_aligned[i] and low[i] <= s2_1M_aligned[i] * 1.005)) and \
               volume_spike[i] and close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects at R1 or R2 with volume and below weekly EMA
            elif ((close[i] < r1_1M_aligned[i] and high[i] >= r1_1M_aligned[i] * 0.995) or
                  (close[i] < r2_1M_aligned[i] and high[i] >= r2_1M_aligned[i] * 0.995)) and \
                 volume_spike[i] and close[i] < ema_21_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S2 or reaches R2
            if close[i] < s2_1M_aligned[i] or close[i] > r2_1M_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R2 or reaches S2
            if close[i] > r2_1M_aligned[i] or close[i] < s2_1M_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals