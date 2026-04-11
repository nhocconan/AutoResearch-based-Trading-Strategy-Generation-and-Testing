#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter
# - Long: Williams %R(14) crosses above -80 (oversold) AND price > 1w EMA(50) (bullish trend)
# - Short: Williams %R(14) crosses below -20 (overbought) AND price < 1w EMA(50) (bearish trend)
# - Exit: Williams %R returns to -50 (mean reversion center)
# - Uses 1d Williams %R for timing, 1w EMA for trend filter to avoid counter-trend trades
# - Works in both bull and bear markets by only taking trades in direction of higher timeframe trend
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits

name = "1d_1w_williamsr_meanreversion_v1"
timeframe = "1d"
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
    
    # Load 1d data ONCE before loop for Williams %R (primary timeframe)
    df_1d = prices  # primary timeframe is 1d, so we can use prices directly
    
    # Load 1w data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1d Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        williams_r_current = williams_r[i]
        williams_r_prev = williams_r[i-1] if i > 0 else williams_r_current
        ema_50_1w_current = ema_50_1w_aligned[i]
        
        # Williams %R signals
        williams_r_oversold = williams_r_current > -80 and williams_r_prev <= -80  # crosses above -80
        williams_r_overbought = williams_r_current < -20 and williams_r_prev >= -20  # crosses below -20
        williams_r_exit = abs(williams_r_current + 50) < 2.5  # near -50 (mean reversion)
        
        # Trend filter: price relative to 1w EMA(50)
        bullish_trend = close_price > ema_50_1w_current
        bearish_trend = close_price < ema_50_1w_current
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: oversold + bullish 1w trend
        if williams_r_oversold and bullish_trend:
            enter_long = True
        
        # Short: overbought + bearish 1w trend
        if williams_r_overbought and bearish_trend:
            enter_short = True
        
        # Exit conditions: mean reversion at Williams %R = -50
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long when Williams %R returns to -50
            exit_long = williams_r_exit
        elif position == -1:
            # Exit short when Williams %R returns to -50
            exit_short = williams_r_exit
        
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