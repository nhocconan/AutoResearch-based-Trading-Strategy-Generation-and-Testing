#/usr/bin/env python3
"""
12h_Triple_Screen_Strategy
12h strategy using Triple Screen method:
- Trend: weekly EMA13 slope (long if rising, short if falling)
- Entry: 12h Stochastic oversold/overbought with volume confirmation
- Exit: opposite stochastic signal or trend change
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA13 for trend
    ema13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_slope = np.diff(ema13_1w, prepend=ema13_1w[0])
    ema13_slope_aligned = align_htf_to_ltf(prices, df_1w, ema13_slope)
    
    # Get daily data for stochastic calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period Stochastic %K
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    stoch_k = 100 * (close_1d - lowest_low) / (highest_high - lowest_low)
    stoch_k = np.where((highest_high - lowest_low) == 0, 50, stoch_k)  # avoid division by zero
    
    # 3-period SMA of %K for %D
    stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values
    
    # Align stochastic to 12h
    stoch_k_aligned = align_htf_to_ltf(prices, df_1d, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d)
    
    # Daily volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # need enough for stochastic
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13_slope_aligned[i]) or np.isnan(stoch_k_aligned[i]) or 
            np.isnan(stoch_d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema13_slope_aligned[i] > 0
        downtrend = ema13_slope_aligned[i] < 0
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_aligned[i]
        
        # Stochastic signals
        stoch_oversold = stoch_k_aligned[i] < 20 and stoch_d_aligned[i] < 20
        stoch_overbought = stoch_k_aligned[i] > 80 and stoch_d_aligned[i] > 80
        
        if position == 0:
            # Long: uptrend + volume + stochastic oversold
            if uptrend and vol_confirm and stoch_oversold:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + stochastic overbought
            elif downtrend and vol_confirm and stoch_overbought:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, overbought stochastic, or volume confirmation
            if not uptrend or stoch_overbought or (vol_confirm and stoch_k_aligned[i] > 50):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, oversold stochastic, or volume confirmation
            if not downtrend or stoch_oversold or (vol_confirm and stoch_k_aligned[i] < 50):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Triple_Screen_Strategy"
timeframe = "12h"
leverage = 1.0