#!/usr/bin/env python3
"""
12h_Weekly_Momentum_Reversal
Hypothesis: Weekly momentum extremes (RSI > 70 or < 30) combined with 12h price rejection 
at weekly support/resistance levels (from prior week's high/low) capture reversal 
opportunities. Works in bull/bear by fading overextended moves. Target: 15-25 trades/year 
(60-100 total over 4 years) to minimize fee drag while capturing high-probability setups.
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
    
    # Weekly data for momentum and levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly RSI (14) for momentum extreme
    rsi_period = 14
    delta = pd.Series(df_1w['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Previous week's high/low for support/resistance
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    
    # Align weekly data to 12h timeframe
    rsi_12h = align_htf_to_ltf(prices, df_1w, rsi_values)
    week_high_12h = align_htf_to_ltf(prices, df_1w, prev_week_high)
    week_low_12h = align_htf_to_ltf(prices, df_1w, prev_week_low)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_12h[i]) or np.isnan(week_high_12h[i]) or 
            np.isnan(week_low_12h[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_12h[i]
        wh = week_high_12h[i]
        wl = week_low_12h[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: RSI < 30 (oversold) + price near weekly low + volume
            if rsi_val < 30 and price <= wl * 1.005 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + price near weekly high + volume
            elif rsi_val > 70 and price >= wh * 0.995 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI > 50 (momentum shift) or price reaches weekly high
            if rsi_val > 50 or price >= wh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI < 50 (momentum shift) or price reaches weekly low
            if rsi_val < 50 or price <= wl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Weekly_Momentum_Reversal"
timeframe = "12h"
leverage = 1.0