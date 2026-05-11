#!/usr/bin/env python3
"""
1d_Camarilla_H4_Trend_Filter
Hypothesis: Daily chart strategy using weekly Camarilla levels (H4/L4) with 1-week trend filter.
Targets mean reversion in range-bound markets and continuation in trending markets.
Uses weekly trend to filter direction for higher probability entries. Designed for low trade frequency (<25/year) to minimize fee drag on 1d timeframe.
Works in bull/bear by adapting to regime: weekly trend determines bias, price action at H4/L4 provides entry.
"""

name = "1d_Camarilla_H4_Trend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Calculate Weekly Camarilla levels (H4/L4) from prior week ===
    # Get weekly OHLC
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior week's OHLC (shifted by 1 to avoid look-ahead)
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    prior_close = df_1w['close'].shift(1).values
    
    # Align to daily timeframe
    prior_high_1d = align_htf_to_ltf(prices, df_1w, prior_high)
    prior_low_1d = align_htf_to_ltf(prices, df_1w, prior_low)
    prior_close_1d = align_htf_to_ltf(prices, df_1w, prior_close)
    
    # Weekly Camarilla H4 and L4 levels (using 1.1 multiplier)
    H4 = prior_close_1d + (prior_high_1d - prior_low_1d) * 1.1 / 2
    L4 = prior_close_1d - (prior_high_1d - prior_low_1d) * 1.1 / 2
    
    # === 1-week EMA20 Trend Filter ===
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_1d = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === Volume Spike Filter (1.5x 20-day EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers EMA and Camarilla calculation)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(H4[i]) or np.isnan(L4[i]) or 
            np.isnan(ema20_1w_1d[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price touches or goes below L4 with uptrend (close > EMA20) and volume spike
            if (low[i] <= L4[i] and 
                close[i] > ema20_1w_1d[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Price touches or goes above H4 with downtrend (close < EMA20) and volume spike
            elif (high[i] >= H4[i] and 
                  close[i] < ema20_1w_1d[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Price touches opposite level or trend reverses
            if position == 1:
                # Exit long if price touches H4 OR trend turns down
                if (high[i] >= H4[i] or close[i] < ema20_1w_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short if price touches L4 OR trend turns up
                if (low[i] <= L4[i] or close[i] > ema20_1w_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals