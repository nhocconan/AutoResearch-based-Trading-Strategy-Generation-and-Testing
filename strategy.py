#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter and volume confirmation.
# In low volatility (BB width < 20th percentile), breakouts have higher probability.
# Trend filter: price > 1d EMA50 for longs, price < 1d EMA50 for shorts.
# Volume > 2x average confirms breakout. Target: 75-200 trades over 4 years.
# Position size: 0.25. Works in bull/bear via volatility contraction/expansion cycle.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Bollinger Bands
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2) on 4h
    close_4h = df_4h['close'].values
    bb_middle = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).rank(pct=True).values
    
    # Calculate 1-day EMA (50-period) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation using 4h volume
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_4h, bb_width_percentile)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_width_percentile_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(vol_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_current = vol_4h_aligned[i]
        
        if position == 0:
            # Volatility contraction: BB width < 20th percentile (squeeze)
            if bb_width_percentile_aligned[i] < 0.2:
                # Enter long: price breaks above upper BB + volume spike + price > 1d EMA50 (uptrend)
                if (price_close > bb_upper_aligned[i] and
                    vol_current > 2.0 * vol_ma_20_4h_aligned[i] and
                    price_close > ema_50_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below lower BB + volume spike + price < 1d EMA50 (downtrend)
                elif (price_close < bb_lower_aligned[i] and
                      vol_current > 2.0 * vol_ma_20_4h_aligned[i] and
                      price_close < ema_50_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit: price returns to middle BB or volatility expands significantly
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below middle BB or BB width > 80th percentile (high volatility)
                if (price_close < bb_middle[i]) or (bb_width_percentile_aligned[i] > 0.8):
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above middle BB or BB width > 80th percentile
                if (price_close > bb_middle[i]) or (bb_width_percentile_aligned[i] > 0.8):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Squeeze_Breakout_1dEMA50_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0