#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_donchian_breakout_v1
# Uses weekly Donchian channel breakouts (20-period) as primary signal,
# confirmed by daily volume and weekly trend (price above/below weekly SMA50).
# Long when price breaks above weekly Donchian high with volume confirmation
# and weekly trend up (price > weekly SMA50).
# Short when price breaks below weekly Donchian low with volume confirmation
# and weekly trend down (price < weekly SMA50).
# Exits when price returns to weekly Donchian middle (mean reversion).
# Designed for low trade frequency (target: 15-25 trades/year) to minimize fee drag.
# Works in trending markets via breakouts and in ranging markets via mean reversion.

name = "6h_1d_1w_donchian_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Weekly SMA50 for trend filter
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1w, donch_mid)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Daily volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(sma50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation for new entries
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above weekly Donchian high with weekly uptrend
        if (close[i] > donch_high_aligned[i] and 
            close_1w[-1] > sma50_1w[-1] if len(close_1w) > 0 else False and  # current weekly close > SMA50
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below weekly Donchian low with weekly downtrend
        elif (close[i] < donch_low_aligned[i] and 
              close_1w[-1] < sma50_1w[-1] if len(close_1w) > 0 else False and  # current weekly close < SMA50
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: price returns to weekly Donchian middle (mean reversion)
        elif position == 1 and close[i] <= donch_mid_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= donch_mid_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals