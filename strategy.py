#!/usr/bin/env python3
"""
1d_1w_RVOL_Momentum_Reversion
Hypothesis: On daily timeframe, use weekly RVOL (relative volume) to detect institutional interest.
Go long when price pulls back to VWAP during high RVOL uptrend, short when rallies to VWAP during high RVOL downtrend.
Weekly trend filter avoids counter-trend trades. Designed for 1-3 trades per month per symbol.
Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 12-36 total trades over 4 years (3-9/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RVOL_Momentum_Reversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR TREND AND RVOL ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Weekly VWAP (typical price * volume cumulative)
    typical_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_1w = (np.cumsum(typical_1w * volume_1w) / np.cumsum(volume_1w))
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Weekly RVOL: current volume / 20-period average volume
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    rvol_1w = volume_1w / vol_ma_1w
    rvol_1w_aligned = align_htf_to_ltf(prices, df_1w, rvol_1w)
    
    # Daily VWAP for entry
    typical = (high + low + close) / 3.0
    vwap = np.cumsum(typical * volume) / np.cumsum(volume)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(rvol_1w_aligned[i]) or np.isnan(vwap[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        trend_up = close_1w_aligned[i] > ema20_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema20_1w_aligned[i]
        
        # Entry conditions: price touches daily VWAP during high RVOL + weekly trend
        # Define touch as within 0.5% of VWAP
        vwap_distance = abs(close[i] - vwap[i]) / vwap[i]
        vwap_touch = vwap_distance < 0.005
        high_rvol = rvol_1w_aligned[i] > 1.8  # 80% above average volume
        
        long_signal = vwap_touch and trend_up and high_rvol and close[i] <= vwap[i]
        short_signal = vwap_touch and trend_down and high_rvol and close[i] >= vwap[i]
        
        # Exit conditions: opposite VWAP touch or trend reversal
        exit_long = (position == 1 and 
                    (vwap_touch and close[i] >= vwap[i]) or not trend_up)
        exit_short = (position == -1 and 
                     (vwap_touch and close[i] <= vwap[i]) or not trend_down)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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