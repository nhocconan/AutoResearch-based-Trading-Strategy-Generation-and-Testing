#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot level bounce with 1-day trend filter and volume confirmation.
# The Camarilla pivot levels (based on previous day's range) provide strong support/resistance levels.
# In ranging markets, price tends to revert to the mean (pivot), while in trending markets,
# breaks of key levels (H3/L3) can signal continuation. The 1-day EMA(50) filters for the dominant trend,
# and volume > 1.3x the 20-period average confirms institutional participation.
# This strategy aims for 15-30 trades per year per symbol (60-120 total over 4 years),
# staying within the optimal range to minimize fee drift while capturing both mean reversion and breakout opportunities.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:  # Need at least 2 days for pivot calculation
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # H4 = Close + 1.5*(High - Low)
    # H3 = Close + 1.1*(High - Low)
    # H2 = Close + 0.6*(High - Low)
    # H1 = Close + 0.318*(High - Low)
    # L1 = Close - 0.318*(High - Low)
    # L2 = Close - 0.6*(High - Low)
    # L3 = Close - 1.1*(High - Low)
    # L4 = Close - 1.5*(High - Low)
    
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate pivot levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.1 * (prev_high - prev_low)
    L3 = prev_close - 1.1 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # 1-day EMA(50) for trend filter
    ema_len = 50
    if len(df_1d) < ema_len:
        return np.zeros(n)
    
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Need EMA and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1-day EMA50
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price touches or goes below L3 (strong support) + above EMA + volume
            if (close[i] <= L3_aligned[i] and 
                above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price touches or goes above H3 (strong resistance) + below EMA + volume
            elif (close[i] >= H3_aligned[i] and 
                  below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to EMA or breaks above H3 (failed support)
            if close[i] >= ema_1d_aligned[i] or close[i] >= H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to EMA or breaks below L3 (failed resistance)
            if close[i] <= ema_1d_aligned[i] or close[i] <= L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_EMA_Volume_v1"
timeframe = "12h"
leverage = 1.0