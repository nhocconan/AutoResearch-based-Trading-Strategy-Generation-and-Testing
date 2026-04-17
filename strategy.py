#!/usr/bin/env python3
"""
1d_WeeklyKeltner_SqueezeBreakout
Strategy: Daily price breaks out of Keltner Channel (20, 2) after weekly Bollinger Band squeeze (BBW < 50th percentile).
Long: Close > KC upper + weekly BBW < 50th percentile
Short: Close < KC lower + weekly BBW < 50th percentile
Exit: Close crosses back through 20-day EMA
Position size: 0.25
Uses weekly volatility contraction (squeeze) to anticipate expansion, KC breakout for direction.
Works in bull/bear: Squeeze filters low-volatility environments, breakout captures resulting volatility expansion.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Bollinger Band width (squeeze detection)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Calculate Bollinger Bands (20, 2) on weekly
    basis_weekly = pd.Series(close_weekly).rolling(window=20, min_periods=20).mean().values
    dev_weekly = 2 * pd.Series(close_weekly).rolling(window=20, min_periods=20).std().values
    upper_weekly = basis_weekly + dev_weekly
    lower_weekly = basis_weekly - dev_weekly
    bb_width_weekly = ((upper_weekly - lower_weekly) / basis_weekly) * 100
    
    # Calculate 50th percentile of BB width (median over lookback)
    bb_width_median = pd.Series(bb_width_weekly).rolling(window=50, min_periods=50).median().values
    squeeze_condition = bb_width_weekly < bb_width_median  # BBW below median = squeeze
    
    # Align squeeze to daily
    squeeze_aligned = align_htf_to_ltf(prices, df_weekly, squeeze_condition)
    
    # Calculate Keltner Channel (20, 2) on daily
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    kc_basis = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=20, min_periods=20).mean().values
    atr[0] = high[0] - low[0]  # first value
    kc_upper = kc_basis + 2 * atr
    kc_lower = kc_basis - 2 * atr
    
    # Calculate 20-day EMA for exit
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute for efficiency
    for i in range(20, n):  # warmup for KC(20)
        # Skip if any required data is not available
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_20[i]) or np.isnan(squeeze_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: KC breakout + weekly squeeze
        breakout_up = close[i] > kc_upper[i]
        breakout_down = close[i] < kc_lower[i]
        
        if position == 0:
            # Long: breakout up + weekly squeeze
            if breakout_up and squeeze_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout down + weekly squeeze
            elif breakout_down and squeeze_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 20-day EMA
            if close[i] < ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 20-day EMA
            if close[i] > ema_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyKeltner_SqueezeBreakout"
timeframe = "1d"
leverage = 1.0