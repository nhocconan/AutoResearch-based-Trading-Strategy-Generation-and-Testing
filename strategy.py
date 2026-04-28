#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and Bollinger Band squeeze confirmation.
# Williams %R(14) < -80 for long, > -20 for short in ranging markets (Bollinger Band Width < 50th percentile).
# 1d EMA34 provides trend filter: only long when price > EMA34, short when price < EMA34.
# Bollinger Band squeeze (low volatility) increases mean reversion edge.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn. Target: 12-37 trades/year.

name = "6h_WilliamsR_MeanReversion_1dEMA34_BBSqueeze_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for EMA34 trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Bollinger Bands (20, 2) for squeeze detection
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2.0 * std_20_1d
    lower_bb_1d = sma_20_1d - 2.0 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d * 100.0  # Percent width
    
    # Calculate Bollinger Band Width percentile (50th = median)
    bb_width_percentile = pd.Series(bb_width_1d).rolling(window=50, min_periods=50).quantile(0.50).values
    bb_squeeze = bb_width_1d < bb_width_percentile  # Low volatility regime
    
    # Align HTF indicators to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bb_squeeze_aligned = align_htf_to_ltf(prices, df_1d, bb_squeeze.astype(float))
    
    # Calculate 6h Williams %R (14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(bb_squeeze_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80.0
        overbought = williams_r[i] > -20.0
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Bollinger Band squeeze (low volatility mean reversion regime)
        in_squeeze = bb_squeeze_aligned[i] > 0.5
        
        long_entry = oversold and price_above_ema and in_squeeze
        short_entry = overbought and price_below_ema and in_squeeze
        
        # Exit conditions: Williams %R reverts to mean (-50 center)
        long_exit = williams_r[i] > -50.0
        short_exit = williams_r[i] < -50.0
        
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