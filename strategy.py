#!/usr/bin/env python3
"""
4h_1d_Camarilla_Reversal_V1
Hypothesis: Combines daily Camarilla pivot levels with 4h RSI reversal signals.
Long when price touches Camarilla L3/S1 level and RSI<30 on 4h; short when price touches H3/S2 level and RSI>70.
Uses Camarilla levels as institutional support/resistance and RSI for exhaustion.
Designed for low trade frequency by requiring price at specific pivot levels + momentum confirmation.
Works in bull via buying dips at support, in bear via selling rallies at resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Reversal_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    daily_range = high_1d - low_1d
    camarilla_h4 = close_1d + 1.5 * daily_range
    camarilla_h3 = close_1d + 1.1 * daily_range
    camarilla_h2 = close_1d + 0.6 * daily_range
    camarilla_h1 = close_1d + 0.3 * daily_range
    camarilla_l1 = close_1d - 0.3 * daily_range
    camarilla_l2 = close_1d - 0.6 * daily_range
    camarilla_l3 = close_1d - 1.1 * daily_range
    camarilla_l4 = close_1d - 1.5 * daily_range
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === 4H DATA FOR RSI ===
    # Calculate RSI on 4h close prices
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Use Wilder's smoothing (alpha = 1/period)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Price proximity to Camarilla levels (within 0.5% tolerance)
        price = close[i]
        h3_level = h3_aligned[i]
        l3_level = l3_aligned[i]
        
        near_h3 = abs(price - h3_level) / h3_level < 0.005  # 0.5% tolerance
        near_l3 = abs(price - l3_level) / l3_level < 0.005  # 0.5% tolerance
        
        # RSI conditions
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        # Entry conditions
        long_setup = near_l3 and rsi_oversold
        short_setup = near_h3 and rsi_overbought
        
        # Exit when price moves away from level or RSI normalizes
        exit_long = not (near_l3 and rsi_oversold)
        exit_short = not (near_h3 and rsi_overbought)
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals