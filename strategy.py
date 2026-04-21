#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Daily KAMA trend filter with weekly Bollinger Band squeeze and volume confirmation.
# Uses weekly Bollinger Band width percentile to detect low volatility squeeze (breakout imminent).
# KAMA (10-period) determines trend direction on daily timeframe.
# Long when: KAMA rising, BB width < 20th percentile (squeeze), volume > 1.5x 20-day average.
# Short when: KAMA falling, BB width < 20th percentile (squeeze), volume > 1.5x 20-day average.
# Exit when KAMA reverses direction or volatility expands (BB width > 80th percentile).
# Target: 15-25 trades/year by requiring squeeze + trend alignment + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) - 10 period
    close = prices['close'].values
    direction = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate weekly Bollinger Band width (20, 2)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    bb_middle = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Align BB width to daily timeframe
    bb_width_aligned = align_htf_to_ltf(prices, df_1w, bb_width)
    
    # Calculate 20th and 80th percentiles of BB width (using expanding window for realism)
    bb_width_percentile_20 = np.full_like(bb_width_aligned, np.nan)
    bb_width_percentile_80 = np.full_like(bb_width_aligned, np.nan)
    
    for i in range(20, len(bb_width_aligned)):
        if not np.isnan(bb_width_aligned[i]):
            hist_width = bb_width_aligned[max(0, i-252):i+1]  # ~1 year lookback
            if len(hist_width) >= 20:
                bb_width_percentile_20[i] = np.percentile(hist_width, 20)
                bb_width_percentile_80[i] = np.percentile(hist_width, 80)
    
    # Daily volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(bb_width_percentile_20[i]) or np.isnan(bb_width_percentile_80[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Bollinger Band squeeze condition: width < 20th percentile
        is_squeeze = bb_width_aligned[i] < bb_width_percentile_20[i]
        
        # Volatility expansion exit: width > 80th percentile
        is_expansion = bb_width_aligned[i] > bb_width_percentile_80[i]
        
        # KAMA trend direction
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        if position == 0:
            # Look for squeeze breakout in direction of KAMA trend
            if is_squeeze and volume_confirm:
                if kama_rising:
                    signals[i] = 0.25
                    position = 1
                elif kama_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if KAMA turns down or volatility expands
                if kama_falling or is_expansion:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if KAMA turns up or volatility expands
                if kama_rising or is_expansion:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_WeeklyBB_Squeeze_Volume"
timeframe = "1d"
leverage = 1.0