#!/usr/bin/env python3
"""
4h_Camarilla_R4_S4_Breakout_12hEMA50_VolumeSqueeze
Hypothesis: Combines tight volatility (Bollinger Band squeeze) with breakouts at extreme Camarilla R4/S4 levels (1.5x range) filtered by 12h EMA50 trend. Volatility squeeze reduces false breakouts, while extreme levels increase reward/risk. Works in bull/bear by following 12h trend. Targets 15-25 trades/year via very strict conditions: requires volatility contraction AND expansion AND trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels (R4/S4: 1.5x range)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_ = df_1d['high'] - df_1d['low']
    R4 = typical_price + (range_ * 1.5 / 2)
    S4 = typical_price - (range_ * 1.5 / 2)
    
    # Align Camarilla levels to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4.values)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4.values)
    
    # Volatility squeeze: Bollinger Band width < 20th percentile (40-period lookback)
    bb_period = 40
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    bb_width = (upper_bb - lower_bb) / sma
    
    # Squeeze condition: BB width below 20th percentile of its own history
    bb_width_pct = pd.Series(bb_width).rolling(window=100, min_periods=100).quantile(0.20).values
    squeeze = bb_width < bb_width_pct
    
    # Breakout condition: price breaks above/below Bollinger Bands (volatility expansion)
    breakout_up = close > upper_bb
    breakout_down = close < lower_bb
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 100)  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(R4_aligned[i]) or
            np.isnan(S4_aligned[i]) or
            np.isnan(squeeze[i]) or
            np.isnan(breakout_up[i]) or
            np.isnan(breakout_down[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Entry: volatility squeeze breakout in trend direction at Camarilla extremes
        long_entry = breakout_up[i] and squeeze[i-1] and close[i] > R4_aligned[i] and uptrend
        short_entry = breakout_down[i] and squeeze[i-1] and close[i] < S4_aligned[i] and downtrend
        
        # Exit: return to midpoint of Bollinger Bands (mean reversion after expansion)
        midpoint_bb = (upper_bb[i] + lower_bb[i]) / 2
        long_exit = close[i] < midpoint_bb
        short_exit = close[i] > midpoint_bb
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R4_S4_Breakout_12hEMA50_VolumeSqueeze"
timeframe = "4h"
leverage = 1.0