#!/usr/bin/env python3
# 1d_camarilla_pivot_v9
# Hypothesis: 1d Camarilla pivot levels (L3/H3, L4/H4) act as intraday support/resistance.
# Enter long at L3 with volume confirmation, short at H3 with volume confirmation.
# Use 1w EMA(34) as regime filter: only long when price > 1w EMA, short when price < 1w EMA.
# Exit on opposite Camarilla level touch (L4 for long exit, H4 for short exit) or EMA crossover.
# Target: 15-25 trades/year. Works in bull/bear: 1w EMA filters trend direction, Camarilla provides precise entries/exits, volume avoids false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_v9"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA(34)
    ema_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    # Need previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous day
    
    # Camarilla levels
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # L4 = close - 1.5 * (high - low)
    hl_range = prev_high - prev_low
    h4 = prev_close + 1.5 * hl_range
    h3 = prev_close + 1.125 * hl_range
    l3 = prev_close - 1.125 * hl_range
    l4 = prev_close - 1.5 * hl_range
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(h4[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(l4[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches L4 (stop) OR H4 (target) OR EMA turns bearish
            if low[i] <= l4[i] or high[i] >= h4[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches H4 (stop) OR L4 (target) OR EMA turns bullish
            if high[i] >= h4[i] or low[i] <= l4[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long entry: price touches L3 AND above 1w EMA (bullish regime)
                if low[i] <= l3[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches H3 AND below 1w EMA (bearish regime)
                elif high[i] >= h3[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals