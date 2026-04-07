#!/usr/bin/env python3
"""
4h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels from weekly timeframe identify key support/resistance levels.
Price rejection at these levels with volume confirmation and weekly trend filter provides
high-probability reversal entries. Works in both bull and bear markets by fading extremes
at statistically significant levels. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1w_trend_volume_v1"
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
    
    # Weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for weekly timeframe
    # Using previous week's OHLC
    wk_high = df_1w['high'].shift(1)
    wk_low = df_1w['low'].shift(1)
    wk_close = df_1w['close'].shift(1)
    
    # Typical price for pivot calculation
    wk_typical = (wk_high + wk_low + wk_close) / 3
    wk_range = wk_high - wk_low
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + 1.5 * range * 1.1
    # H3 = close + 1.25 * range * 1.1
    # H2 = close + 1.166 * range * 1.1
    # H1 = close + 1.083 * range * 1.1
    # L1 = close - 1.083 * range * 1.1
    # L2 = close - 1.166 * range * 1.1
    # L3 = close - 1.25 * range * 1.1
    # L4 = close - 1.5 * range * 1.1
    
    camarilla_h4 = wk_close + 1.5 * wk_range * 1.1
    camarilla_h3 = wk_close + 1.25 * wk_range * 1.1
    camarilla_h2 = wk_close + 1.166 * wk_range * 1.1
    camarilla_h1 = wk_close + 1.083 * wk_range * 1.1
    camarilla_l1 = wk_close - 1.083 * wk_range * 1.1
    camarilla_l2 = wk_close - 1.166 * wk_range * 1.1
    camarilla_l3 = wk_close - 1.25 * wk_range * 1.1
    camarilla_l4 = wk_close - 1.5 * wk_range * 1.1
    
    # Weekly EMA for trend filter (20-period)
    ema_20 = df_1w['close'].ewm(span=20, adjust=False).mean()
    
    # Align all weekly data to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4.values)
    h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3.values)
    h2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h2.values)
    h1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h1.values)
    l1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l1.values)
    l2_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l2.values)
    l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3.values)
    l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4.values)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20.values)
    
    # Volume confirmation (20-period average = ~10 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches H3 or trend turns bearish
            if close[i] >= h3_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches L3 or trend turns bullish
            if close[i] <= l3_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price rejects L3/L4 with volume and bullish weekly trend
            if (close[i] <= l3_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                # Additional confirmation: price closing above L3
                if i > 0 and close[i] > low[i]:
                    position = 1
                    signals[i] = 0.25
            # Short entry: price rejects H3/H4 with volume and bearish weekly trend
            elif (close[i] >= h3_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                # Additional confirmation: price closing below H3
                if i > 0 and close[i] < high[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals