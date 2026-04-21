#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (price > 1d EMA),
# we fade extremes: short when %R > -20 (overbought), long when %R < -80 (oversold).
# Volume > 1.3x average confirms momentum. Target: 50-150 total trades over 4 years.
# Position size: 0.25 to manage risk during drawdowns.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 14:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R (14-period) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = ((highest_high - close_6h) / (highest_high - lowest_low)) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align Williams %R to lower timeframe (6h -> 6f alignment is identity, but keep for consistency)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Calculate 1-day EMA (50-period) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation using 6h volume
    vol_6h = df_6h['volume'].values
    vol_ma_20_6h = pd.Series(vol_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_6h_current = align_htf_to_ltf(prices, df_6h, vol_6h)[i]
        
        if position == 0:
            # Enter long: Williams %R oversold (< -80) + volume surge + price > 1d EMA (uptrend bias)
            if (williams_r_aligned[i] < -80 and
                vol_6h_current > 1.3 * vol_ma_20_6h_aligned[i] and
                price_close > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought (> -20) + volume surge + price < 1d EMA (downtrend bias)
            elif (williams_r_aligned[i] > -20 and
                  vol_6h_current > 1.3 * vol_ma_20_6h_aligned[i] and
                  price_close < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Williams %R returns to neutral zone (-50) or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: %R > -50 (leaving oversold) or trend turns down
                if (williams_r_aligned[i] > -50) or (price_close < ema_50_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: %R < -50 (leaving overbought) or trend turns up
                if (williams_r_aligned[i] < -50) or (price_close > ema_50_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA50_Volume_Trend"
timeframe = "6h"
leverage = 1.0