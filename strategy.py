#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter
    # Long when: BB width < 20th percentile (squeeze) AND price breaks above upper band AND 12h EMA20 > EMA50 (uptrend)
    # Short when: BB width < 20th percentile (squeeze) AND price breaks below lower band AND 12h EMA20 < EMA50 (downtrend)
    # Exit when: price returns to middle band (mean reversion) OR adverse 12h EMA crossover
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 12h EMA trend filter preventing counter-trend trades during squeezes.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate 12h EMA20 and EMA50
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Bollinger Bands on 6h (20, 2)
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_bb + (bb_std * std_bb)
    lower_band = sma_bb - (bb_std * std_bb)
    middle_band = sma_bb
    bb_width = (upper_band - lower_band) / middle_band
    
    # Calculate 20th percentile of BB width for squeeze detection (using expanding window)
    bb_width_percentile = pd.Series(bb_width).expanding(min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(sma_bb[i]) or np.isnan(std_bb[i]) or 
            np.isnan(bb_width[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(ema20_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Bollinger Band conditions
        squeeze = squeeze_condition[i]
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        return_to_middle = (abs(close[i] - middle_band[i]) < 0.1 * std_bb[i])  # Within 10% of middle band
        
        # 12h EMA trend filter
        uptrend_12h = ema20_12h_aligned[i] > ema50_12h_aligned[i]
        downtrend_12h = ema20_12h_aligned[i] < ema50_12h_aligned[i]
        
        # Entry conditions (require squeeze breakout)
        long_entry = squeeze and breakout_up and uptrend_12h and position != 1
        short_entry = squeeze and breakout_down and downtrend_12h and position != -1
        
        # Exit conditions
        exit_long = return_to_middle or (position == 1 and not uptrend_12h)
        exit_short = return_to_middle or (position == -1 and not downtrend_12h)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
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

name = "6h_12h_bb_squeeze_breakout_trend_v1"
timeframe = "6h"
leverage = 1.0