#!/usr/bin/env python3
"""
6h_12h_Squeeze_Breakout_Volume
Hypothesis: Bollinger Band squeeze on 12h timeframe identifies low volatility periods.
Breakout from squeeze with volume expansion and 6h momentum captures explosive moves.
Works in bull markets (upside breakouts) and bear markets (downside breakdowns).
Low trade frequency (~15-30/year) minimizes fee flood while capturing large moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_Squeeze_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR BOLLINGER BANDS ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper = sma_20 + 2 * std_20
    lower = sma_20 - 2 * std_20
    bb_width = (upper - lower) / sma_20  # normalized width
    
    # Align to 6h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower)
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    
    # === 6H MOMENTUM AND VOLUME ===
    # RSI(14) on 6h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Squeeze condition: low volatility (BB width < 20th percentile of last 50 periods)
        if i >= 50:
            bb_width_past = bb_width_aligned[max(0, i-50):i]
            bb_width_percentile = np.percentile(bb_width_past, 20) if len(bb_width_past) > 0 else 1.0
            squeeze = bb_width_aligned[i] < bb_width_percentile
        else:
            squeeze = False
        
        # Breakout conditions
        breakout_up = close[i] > upper_aligned[i]
        breakout_down = close[i] < lower_aligned[i]
        
        # Momentum filter: RSI > 55 for long, < 45 for short
        mom_long = rsi[i] > 55
        mom_short = rsi[i] < 45
        
        # Volume expansion: vol_ratio > 1.8
        vol_expand = vol_ratio[i] > 1.8
        
        # Entry logic
        if squeeze and breakout_up and mom_long and vol_expand and position != 1:
            position = 1
            signals[i] = 0.25
        elif squeeze and breakout_down and mom_short and vol_expand and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Exit conditions: opposite breakout or loss of momentum
            exit_long = (breakout_down and mom_short) or (rsi[i] < 40 and position == 1)
            exit_short = (breakout_up and mom_long) or (rsi[i] > 60 and position == -1)
            
            if exit_long and position == 1:
                position = 0
                signals[i] = 0.0
            elif exit_short and position == -1:
                position = 0
                signals[i] = 0.0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals