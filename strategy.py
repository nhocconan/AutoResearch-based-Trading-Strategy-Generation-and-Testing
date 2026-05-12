# 6h_WeeklyPivot_Pullback_Trend
# Hypothesis: In trending markets, price pulls back to weekly pivot levels (PP) before resuming trend.
# Weekly PP acts as dynamic support/resistance. In uptrend, buy near PP; in downtrend, sell near PP.
# Uses 1w trend filter (EMA50) and 1d volume spike for confirmation.
# Works in bull (buy pullbacks) and bear (sell rallies) by adapting to weekly trend.
# Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.

#!/usr/bin/env python3
name = "6h_WeeklyPivot_Pullback_Trend"
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
    
    # === Weekly data for pivot and trend ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Pivot Point (PP) = (H + L + C) / 3
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === Daily volume spike filter ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_avg_1d)  # Moderate threshold for 6h
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pp_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend (price > weekly EMA50), pullback to PP, volume spike
            if (close[i] > ema50_1w_aligned[i] and
                abs((close[i] - pp_1w_aligned[i]) / pp_1w_aligned[i]) < 0.005 and  # Within 0.5% of PP
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < weekly EMA50), pullback to PP, volume spike
            elif (close[i] < ema50_1w_aligned[i] and
                  abs((close[i] - pp_1w_aligned[i]) / pp_1w_aligned[i]) < 0.005 and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below PP or trend reverses
            if close[i] < pp_1w_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above PP or trend reverses
            if close[i] > pp_1w_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals