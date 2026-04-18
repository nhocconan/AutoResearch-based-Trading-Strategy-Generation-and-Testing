#!/usr/bin/env python3
"""
6h_Pivot_R1S1_Breakout_Pullback_Volume
Hypothesis: Combines daily Camarilla pivot levels with 6h price action. 
Long when price breaks above R1 then pulls back to R1 with volume confirmation in uptrend.
Short when price breaks below S1 then pulls back to S1 with volume confirmation in downtrend.
Uses 12h EMA34 for trend filter to avoid counter-trend trades.
Target: 20-40 trades/year to minimize fee drag while capturing high-probability pullbacks.
Works in both bull (buying pullbacks in uptrend) and bear (selling pullbacks in downtrend) markets.
"""

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
    
    # Daily data for Camarilla pivots (calculated once)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = close_1d + (range_ * 1.1 / 12)
    s1 = close_1d - (range_ * 1.1 / 12)
    r4 = close_1d + (range_ * 1.1 / 2)
    s4 = close_1d - (range_ * 1.1 / 2)
    
    # Align daily levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    # Track breakout state: 1 = bullish breakout confirmed, -1 = bearish breakout confirmed
    breakout_state = 0
    
    start_idx = max(20, 34)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema34 = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Update breakout state based on daily closes (only when 12h bar closes)
        # We use the 12h close to determine if we should update state
        if i % 2 == 0:  # Every other 6h bar approximates 12h boundary (simplified)
            # In practice, we'd check if current 12h bar just closed
            # For simplicity, we update state when price crosses R1/S4 or S1/R4
            if price > r4_aligned[i]:
                breakout_state = 1  # Bullish breakout
            elif price < s4_aligned[i]:
                breakout_state = -1  # Bearish breakout
        
        if position == 0:
            # Long: price pulls back to R1 after bullish breakout with volume spike
            if (breakout_state == 1 and
                low[i] <= r1 <= high[i] and  # price touches R1
                close[i] > r1 and            # closes above R1 (confirms bounce)
                vol_spike and
                price > ema34):              # uptrend filter
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to S1 after bearish breakout with volume spike
            elif (breakout_state == -1 and
                  low[i] <= s1 <= high[i] and  # price touches S1
                  close[i] < s1 and            # closes below S1 (confirms bounce)
                  vol_spike and
                  price < ema34):              # downtrend filter
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below S1 or trend reverses
            if close[i] < s1 or price < ema34:
                signals[i] = 0.0
                position = 0
                breakout_state = 0  # Reset breakout state on exit
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above R1 or trend reverses
            if close[i] > r1 or price > ema34:
                signals[i] = 0.0
                position = 0
                breakout_state = 0  # Reset breakout state on exit
    
    return signals

name = "6h_Pivot_R1S1_Breakout_Pullback_Volume"
timeframe = "6h"
leverage = 1.0