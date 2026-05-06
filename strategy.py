#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with 1d Supertrend trend filter and volume confirmation
# - Uses Donchian channel breakout for entry signals
# - Uses 1d Supertrend (ATR=10, multiplier=3) to filter for trend direction
# - Requires volume spike (2x 20-period average) for confirmation
# - Exits when price reverses back into the Donchian channel
# - Designed to capture strong trending moves with confirmation to reduce whipsaws
# - Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_DonchianBreakout_Supertrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian Channel (20)
    high_4h = high
    low_4h = low
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Supertrend (ATR=10, multiplier=3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # ATR(10)
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3 * atr_10
    basic_lb = (high_1d + low_1d) / 2 - 3 * atr_10
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    final_ub[0] = basic_ub[0]
    final_lb[0] = basic_lb[0]
    
    for i in range(1, len(basic_ub)):
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros_like(close_1d)
    supertrend[0] = final_ub[0]
    for i in range(1, len(supertrend)):
        if supertrend[i-1] == final_ub[i-1]:
            supertrend[i] = final_lb[i] if close_1d[i] <= final_lb[i] else final_ub[i]
        else:
            supertrend[i] = final_ub[i] if close_1d[i] >= final_ub[i] else final_lb[i]
    
    # Determine trend direction: 1 for uptrend (price above Supertrend), -1 for downtrend
    trend_dir = np.where(close_1d > supertrend, 1, -1)
    
    # Align 1d indicators to 4h timeframe
    trend_dir_4h = align_htf_to_ltf(prices, df_1d, trend_dir)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(trend_dir_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakout with trend alignment and volume confirmation
            bullish_breakout = close[i] > donchian_high[i] and trend_dir_4h[i] == 1
            bearish_breakout = close[i] < donchian_low[i] and trend_dir_4h[i] == -1
            
            if bullish_breakout and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            elif bearish_breakout and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks back below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks back above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals