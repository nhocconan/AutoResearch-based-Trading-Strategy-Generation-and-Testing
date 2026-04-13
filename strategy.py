#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary timeframe with 1w HTF filter
    # Strategy: 6h Williams %R mean reversion within 1w trend
    # Long: 1w uptrend (price > 1w EMA50) + 6h Williams %R < -80 (oversold)
    # Short: 1w downtrend (price < 1w EMA50) + 6h Williams %R > -20 (overbought)
    # Exit: Williams %R returns to -50 level
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # Williams %R is effective at catching reversals in trending markets
    # Using 1w EMA50 for trend filter avoids whipsaws in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Williams %R on 6h data (14-period)
    def calculate_williams_r(high, low, close, window=14):
        highest_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
        williams_r = np.where(
            (highest_high - lowest_low) != 0,
            -100 * (highest_high - close) / (highest_high - lowest_low),
            -50  # default when no range
        )
        return williams_r
    
    williams_r = calculate_williams_r(high, low, close, window=14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):  # start from 60 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w uptrend/downtrend
        is_uptrend = close[i] > ema50_1w_aligned[i]
        is_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Williams %R conditions
        is_oversold = williams_r[i] < -80
        is_overbought = williams_r[i] > -20
        is_exit = abs(williams_r[i] + 50) < 5  # Near -50 level
        
        # Entry conditions
        enter_long = is_uptrend and is_oversold
        enter_short = is_downtrend and is_overbought
        
        # Exit conditions
        exit_long = position == 1 and is_exit
        exit_short = position == -1 and is_exit
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1w_williamsr_meanreversion_v1"
timeframe = "6h"
leverage = 1.0