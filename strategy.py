#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1S1_MeanReversion_Plus_Breakout_Filter
Hypothesis: In 12h timeframe, price tends to mean-revert from daily S1/R1 levels with volume confirmation, but breaks through S4/R4 indicate strong momentum. Uses strict entry conditions to limit trades (target: 12-37/year). Works in bull/bear markets: fades extremes in range, captures breakouts in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot points
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily Camarilla pivot levels
    P = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    r1_daily = P + (range_daily * 0.382)
    s1_daily = P - (range_daily * 0.382)
    r4_daily = P + (range_daily * 1.5000)
    s4_daily = P - (range_daily * 1.5000)
    
    # Align daily Camarilla levels to 12h timeframe
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r4_daily_aligned = align_htf_to_ltf(prices, df_daily, r4_daily)
    s4_daily_aligned = align_htf_to_ltf(prices, df_daily, s4_daily)
    
    # Main timeframe data (12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 24-period average (24*12h = 12 days)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.8 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if NaN in critical values
        if (np.isnan(r1_daily_aligned[i]) or np.isnan(s1_daily_aligned[i]) or 
            np.isnan(r4_daily_aligned[i]) or np.isnan(s4_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r4 = r4_daily_aligned[i]
        s4 = s4_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Fade at S1/R1: mean reversion from daily support/resistance with volume
            # Long: price shows rejection of S1 with volume confirmation
            if price > s1 and price < (s1 + (r1 - s1) * 0.25) and vol_ok:
                # Require bullish close (close > midpoint)
                if close[i] > (high[i] + low[i]) / 2:
                    signals[i] = 0.25
                    position = 1
            # Short: price shows rejection of R1 with volume confirmation
            elif price < r1 and price > (r1 - (r1 - s1) * 0.25) and vol_ok:
                # Require bearish close (close < midpoint)
                if close[i] < (high[i] + low[i]) / 2:
                    signals[i] = -0.25
                    position = -1
            # Breakout at S4/R4: strong momentum with volume
            elif price > r4 and vol_ok:
                signals[i] = 0.25
                position = 1
            elif price < s4 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to S1 (mean reversion) or breaks below S4 (failed breakout)
            if price < s1 or price < s4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to R1 (mean reversion) or breaks above R4 (failed breakdown)
            if price > r1 or price > r4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Camarilla_R1S1_MeanReversion_Plus_Breakout_Filter"
timeframe = "12h"
leverage = 1.0