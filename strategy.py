#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_1wTrend_v1
Hypothesis: Price often reacts to Camarilla pivot levels (R1/S1) on 12h timeframe. 
Breakouts above R1 or below S1 with volume confirmation and aligned weekly trend 
capture strong directional moves. Weekly trend filter avoids counter-trend trades 
in choppy markets, improving win rate. Designed for low trade frequency (<30/year) 
to minimize fee drag while capturing explosive moves in both bull and bear markets.
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
    
    # Calculate Camarilla levels for 12h timeframe (using previous 12h bar's H/L/C)
    # Since we don't have previous bar's OHLC directly, we'll calculate from current bar's data
    # using a rolling window approach that mimics previous bar's H/L/C
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Use previous bar's H/L/C (shift by 1 to avoid look-ahead)
    prev_high = high_series.shift(1)
    prev_low = low_series.shift(1)
    prev_close = close_series.shift(1)
    
    # Camarilla calculations
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    r1 = R1.values
    s1 = S1.values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly EMA34 trend filter (using weekly data)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 21  # Need Camarilla and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or 
            np.isnan(s1[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_spike = volume_spike[i]
        ema_1w_val = ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and above weekly EMA34
            if price > r1_val and vol_spike and price > ema_1w_val:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and below weekly EMA34
            elif price < s1_val and vol_spike and price < ema_1w_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below S1 or below weekly EMA34
            if price < s1_val or price < ema_1w_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above R1 or above weekly EMA34
            if price > r1_val or price > ema_1w_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_With_Volume_and_1wTrend_v1"
timeframe = "12h"
leverage = 1.0