#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversals with 1d trend filter
# - Williams %R(14) measures overbought/oversold levels (-100 to 0)
# - Extreme readings below -90 (oversold) or above -10 (overbought) signal potential reversals
# - Entries require: 1) Williams %R extreme, 2) price rejection from extreme (pin bar), 3) 1d EMA(50) trend alignment
# - Long: %R < -90 + bullish pin bar (close > open + (high-low)*0.6) + price > 1d EMA(50)
# - Short: %R > -10 + bearish pin bar (close < open - (high-low)*0.6) + price < 1d EMA(50)
# - Exit: opposite Williams %R extreme or price crosses 1d EMA(50)
# - Target: 12-25 trades/year (50-100 total over 4 years) to minimize fee drag
# - Works in both bull/bear markets by capturing exhaustion reversals at extremes

name = "6h_1d_williamsr_extreme_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for EMA trend filter (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Pre-compute 6h Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        open_price_i = open_price[i]
        high_price = high[i]
        low_price = low[i]
        
        # Williams %R levels
        wr_value = williams_r[i]
        
        # Pin bar detection: strong rejection from extremes
        body_size = abs(close_price - open_price_i)
        total_range = high_price - low_price
        is_bullish_pin = (close_price > open_price_i) and (body_size > 0.6 * total_range) and (total_range > 0)
        is_bearish_pin = (close_price < open_price_i) and (body_size > 0.6 * total_range) and (total_range > 0)
        
        # 1d EMA trend bias
        ema_bias_long = close_price > ema_50_1d_aligned[i]
        ema_bias_short = close_price < ema_50_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long entry: extreme oversold (%R < -90) + bullish pin bar + long-term uptrend
        if wr_value < -90 and is_bullish_pin and ema_bias_long:
            enter_long = True
        
        # Short entry: extreme overbought (%R > -10) + bearish pin bar + long-term downtrend
        if wr_value > -10 and is_bearish_pin and ema_bias_short:
            enter_short = True
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Williams %R reaches extreme overbought or price crosses below 1d EMA
            exit_long = (wr_value > -10) or (close_price < ema_50_1d_aligned[i])
        elif position == -1:
            # Exit short if Williams %R reaches extreme oversold or price crosses above 1d EMA
            exit_short = (wr_value < -90) or (close_price > ema_50_1d_aligned[i])
        
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