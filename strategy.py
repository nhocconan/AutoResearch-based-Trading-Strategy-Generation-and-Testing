#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with weekly volatility filter and daily mean-reversion
# Uses weekly Bollinger Bandwidth percentile to identify low volatility regimes
# Enters mean-reversion trades at daily Bollinger Band extremes during low volatility
# Exits at Bollinger middle band. Designed for 20-40 trades/year with low turnover.
# Weekly filter avoids choppy markets, improving win rate and reducing false signals.

name = "4h_1w_bb_width_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bandwidth (20-period)
    close_1w = df_1w['close'].values
    sma_20 = np.full_like(close_1w, np.nan)
    std_20 = np.full_like(close_1w, np.nan)
    
    for i in range(20, len(close_1w)):
        sma_20[i] = np.mean(close_1w[i-20:i+1])
        std_20[i] = np.std(close_1w[i-20:i+1])
    
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Weekly Bollinger Bandwidth percentile (50-period lookback)
    bb_width_percentile = np.full_like(bb_width, np.nan)
    for i in range(50, len(bb_width)):
        window = bb_width[i-50:i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_width_percentile[i] = (np.sum(valid <= bb_width[i]) / len(valid)) * 100
    
    # Low volatility regime: BB width percentile < 30 (squeeze)
    low_vol_regime = bb_width_percentile < 30
    low_vol_aligned = align_htf_to_ltf(prices, df_1w, low_vol_regime)
    
    # Daily Bollinger Bands (20-period) for entry/exit
    sma_20_d = np.full_like(close, np.nan)
    std_20_d = np.full_like(close, np.nan)
    
    for i in range(20, len(close)):
        sma_20_d[i] = np.mean(close[i-20:i+1])
        std_20_d[i] = np.std(close[i-20:i+1])
    
    upper_bb_d = sma_20_d + 2 * std_20_d
    lower_bb_d = sma_20_d - 2 * std_20_d
    middle_bb_d = sma_20_d
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if volatility filter not ready
        if np.isnan(low_vol_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Only trade in low volatility regimes
        if not low_vol_aligned[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Mean reversion signals at Bollinger extremes
        if np.isnan(upper_bb_d[i]) or np.isnan(lower_bb_d[i]) or np.isnan(middle_bb_d[i]):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Long when price touches lower BB, short when touches upper BB
        long_signal = low[i] <= lower_bb_d[i]
        short_signal = high[i] >= upper_bb_d[i]
        
        # Exit when price returns to middle BB
        exit_long = position == 1 and high[i] >= middle_bb_d[i]
        exit_short = position == -1 and low[i] <= middle_bb_d[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals