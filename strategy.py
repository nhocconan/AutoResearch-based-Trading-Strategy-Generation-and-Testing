#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1w trend filter
# Williams %R identifies overbought/oversold conditions, mean reverts in ranging markets
# 1-week EMA filter ensures we only trade in direction of higher timeframe trend
# Works in both bull/bear markets: mean reversion in ranges, trend following in trends
# Target: 50-150 total trades over 4 years (12-37/year)
# Position size: 0.25 (25% of capital)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA on 1-week close (trend filter)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Williams %R on 6h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(14, n):
        # Skip if data not ready
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA20
        above_ema = close[i] > ema_20_1w_aligned[i]
        below_ema = close[i] < ema_20_1w_aligned[i]
        
        # Williams %R conditions: oversold < -80, overbought > -20
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Exit conditions: Williams %R returns to neutral zone (-50)
        exit_long = position == 1 and williams_r[i] > -50
        exit_short = position == -1 and williams_r[i] < -50
        
        # Execute signals
        if oversold and above_ema and position != 1:
            # Long setup: oversold in uptrend
            position = 1
            signals[i] = position_size
        elif overbought and below_ema and position != -1:
            # Short setup: overbought in downtrend
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_williamsr_mean_reversion"
timeframe = "6h"
leverage = 1.0