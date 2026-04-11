#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_Volume
Hypothesis: Uses 1d Camarilla pivot levels with weekly trend filter and volume confirmation.
Trades breakouts of key intraday pivot levels (L3, H3) only in direction of weekly trend.
Designed for low trade frequency (<25/year) with high win rate by combining institutional levels,
trend alignment, and volume confirmation. Works in bull/bear by following weekly trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Pivot_Breakout_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA20 to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla pivot levels using previous day's data
        # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
        #          L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Skip if previous day data is invalid
        if np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate pivot levels
        range_hl = prev_high - prev_low
        h3 = prev_close + 1.1 * range_hl
        l3 = prev_close - 1.1 * range_hl
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > h3  # Break above H3 level
        breakdown_down = close[i] < l3  # Break below L3 level
        
        # Entry conditions: only trade in direction of weekly trend
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: return to opposite Camarilla level or trend reversal
        # Calculate L3 and H3 for exit conditions (same as entry)
        l3_exit = prev_close - 1.1 * range_hl
        h3_exit = prev_close + 1.1 * range_hl
        
        long_exit = (close[i] < l3_exit) or (not uptrend)  # Break below L3 or trend change
        short_exit = (close[i] > h3_exit) or (not downtrend)  # Break above H3 or trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals