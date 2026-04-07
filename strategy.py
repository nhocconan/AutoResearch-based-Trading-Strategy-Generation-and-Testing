#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_volume_v1
# Hypothesis: 12h chart trades on daily Camarilla pivot levels with volume confirmation.
# Daily Camarilla levels (S1/S2/S3/S4 and R1/R2/R3/R4) act as strong support/resistance.
# Price breaking through S3/R3 with volume indicates trend continuation; bouncing from S1/R1 with
# volume indicates mean reversion. Works in both bull (breakouts) and bear (reversals) markets.
# Uses 12h for entries and 1d for context, targeting 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
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
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # H4 = Close + 1.5 * (High - Low)
    # H3 = Close + 1.0 * (High - Low)
    # H2 = Close + 0.5 * (High - Low)
    # H1 = Close + 0.25 * (High - Low)
    # L1 = Close - 0.25 * (High - Low)
    # L2 = Close - 0.5 * (High - Low)
    # L3 = Close - 1.0 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.0 * (prev_high - prev_low)
    H2 = prev_close + 0.5 * (prev_high - prev_low)
    H1 = prev_close + 0.25 * (prev_high - prev_low)
    L1 = prev_close - 0.25 * (prev_high - prev_low)
    L2 = prev_close - 0.5 * (prev_high - prev_low)
    L3 = prev_close - 1.0 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align all levels to 12h timeframe (shifted by 1 day for lookback)
    H4_12h = align_htf_to_ltf(prices, df_1d, H4)
    H3_12h = align_htf_to_ltf(prices, df_1d, H3)
    H2_12h = align_htf_to_ltf(prices, df_1d, H2)
    H1_12h = align_htf_to_ltf(prices, df_1d, H1)
    L1_12h = align_htf_to_ltf(prices, df_1d, L1)
    L2_12h = align_htf_to_ltf(prices, df_1d, L2)
    L3_12h = align_htf_to_ltf(prices, df_1d, L3)
    L4_12h = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: volume > 1.5x 30-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any pivot level is not ready
        if (np.isnan(H4_12h[i]) or np.isnan(H3_12h[i]) or np.isnan(H2_12h[i]) or 
            np.isnan(H1_12h[i]) or np.isnan(L1_12h[i]) or np.isnan(L2_12h[i]) or 
            np.isnan(L3_12h[i]) or np.isnan(L4_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L2 (strong support broken)
            if close[i] < L2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H2 (strong resistance broken)
            if close[i] > H2_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume must be present for any entry
            if not volume_spike[i]:
                signals[i] = 0.0
                continue
                
            # Long entry: price breaks above H3 with volume (bullish breakout)
            # OR price bounces from L3/L4 with volume (bullish reversal)
            if ((close[i] > H3_12h[i] and close[i-1] <= H3_12h[i-1]) or  # Breakout above H3
                ((close[i] > L3_12h[i] and close[i-1] <= L3_12h[i-1]) or  # Bounce from L3
                 (close[i] > L4_12h[i] and close[i-1] <= L4_12h[i-1])) and  # Bounce from L4
                close[i] < H2_12h[i]):  # But not above strong resistance
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 with volume (bearish breakout)
            # OR price rejects from H3/H4 with volume (bearish reversal)
            elif ((close[i] < L3_12h[i] and close[i-1] >= L3_12h[i-1]) or  # Breakdown below L3
                  ((close[i] < H3_12h[i] and close[i-1] >= H3_12h[i-1]) or  # Rejection from H3
                   (close[i] < H4_12h[i] and close[i-1] >= H4_12h[i-1])) and  # Rejection from H4
                  close[i] > L2_12h[i]):  # But not below strong support
                position = -1
                signals[i] = -0.25
    
    return signals