#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter
# - Long: Williams %R(14) < -80 (oversold) AND price > 1d EMA50 (uptrend)
# - Short: Williams %R(14) > -20 (overbought) AND price < 1d EMA50 (downtrend)
# - Exit: Williams %R returns to -50 level (mean reversion)
# - Uses 6h Williams %R for timing, 1d EMA50 for trend filter
# - Works in both bull and bear markets by fading extremes in direction of higher timeframe trend
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA50 trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        
        # Williams %R levels
        wr = williams_r[i]
        
        # Trend filter: price vs 1d EMA50
        price_above_ema = close_price > ema50_1d_aligned[i]
        price_below_ema = close_price < ema50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: oversold AND uptrend (price > EMA50)
        if wr < -80 and price_above_ema:
            enter_long = True
        
        # Short: overbought AND downtrend (price < EMA50)
        if wr > -20 and price_below_ema:
            enter_short = True
        
        # Exit conditions: mean reversion at Williams %R = -50
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R returns to -50 (mean reversion)
            exit_long = wr >= -50
        elif position == -1:
            # Exit short when Williams %R returns to -50 (mean reversion)
            exit_short = wr <= -50
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals