#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR(14) volatility filter.
# Uses 1d primary timeframe targeting 7-25 trades/year (30-100 total over 4 years).
# 1w EMA34 provides primary trend filter: bull when close > EMA34, bear when close < EMA34.
# Donchian(20) from 1d provides institutional breakout levels with proven edge.
# ATR(14) > 0.5 * ATR(50) ensures breakouts occur in sufficient volatility regimes.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.

name = "1d_Donchian20_1wEMA34_Trend_ATR_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Donchian and ATR calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian(20) channels
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for ATR50 and EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or
            np.isnan(lowest_20_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w EMA34 direction
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_20_aligned[i]
        short_breakout = close[i] < lowest_20_aligned[i]
        
        # Volatility filter: ATR(14) > 0.5 * ATR(50)
        vol_filter = atr_14_aligned[i] > 0.5 * atr_50_aligned[i]
        
        long_entry = price_above_ema and long_breakout and vol_filter
        short_entry = price_below_ema and short_breakout and vol_filter
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < lowest_20_aligned[i]  # Exit long at lower Donchian
        short_exit = close[i] > highest_20_aligned[i]  # Exit short at upper Donchian
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals