#!/usr/bin/env python3
# 4h_1D_Camarilla_R1_S1_Breakout_Pullback_Reentry
# Hypothesis: Breakouts from daily Camarilla R1/S1 levels with pullback confirmation and re-entry on trend continuation.
# In bull markets: buy pullbacks to R1 in uptrend; in bear markets: sell rallies to S1 in downtrend.
# Uses EMA trend filter and volume confirmation to avoid false breakouts. Targets 25-40 trades/year.

name = "4h_1D_Camarilla_R1_S1_Breakout_Pullback_Reentry"
timeframe = "4h"
leverage = 1.0

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
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma)
    
    # Daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla R1 and S1 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.0 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.0 / 4
    
    # Align daily levels to 4h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Track breakout levels for pullback entry
    breakout_high = np.full(n, np.nan)
    breakout_low = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_active = False  # Track if we have an active breakout level
    
    for i in range(50, n):
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                breakout_active = False
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Check for new breakout
            bullish_breakout = (close[i] > R1_1d_aligned[i] and 
                               volume_confirm[i] and 
                               close[i] > ema_34_1d_aligned[i])
            bearish_breakout = (close[i] < S1_1d_aligned[i] and 
                               volume_confirm[i] and 
                               close[i] < ema_34_1d_aligned[i])
            
            if bullish_breakout:
                breakout_high[i] = R1_1d_aligned[i]  # Store pullback level
                breakout_active = True
                signals[i] = 0.0  # Wait for pullback
            elif bearish_breakout:
                breakout_low[i] = S1_1d_aligned[i]   # Store pullback level
                breakout_active = True
                signals[i] = 0.0  # Wait for pullback
            else:
                signals[i] = 0.0
                breakout_active = False
        
        elif position == 1:
            # Long position: exit on trend reversal or re-entry on pullback
            if close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                breakout_active = False
            else:
                # Look for pullback to breakout level for re-entry
                pullback_level = breakout_high[i]
                if not np.isnan(pullback_level) and low[i] <= pullback_level:
                    # Re-enter on pullback
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25  # Hold position
        
        elif position == -1:
            # Short position: exit on trend reversal or re-entry on pullback
            if close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                breakout_active = False
            else:
                # Look for pullback to breakout level for re-entry
                pullback_level = breakout_low[i]
                if not np.isnan(pullback_level) and high[i] >= pullback_level:
                    # Re-enter on pullback
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25  # Hold position
    
    return signals