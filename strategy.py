#!/usr/bin/env python3
# 1d_Williams_Fractal_1wTrend_Breakout
# Williams fractal breakout on 1d with 1w trend filter
# Long when bullish fractal breaks above resistance and 1w EMA > 1w EMA(34) (bullish trend)
# Short when bearish fractal breaks below support and 1w EMA < 1w EMA(34) (bearish trend)
# Exit when opposite fractal breaks or trend reverses
# Position size: 0.25 to manage drawdown
# Williams fractals require 2-bar confirmation, so use additional_delay_bars=2

name = "1d_Williams_Fractal_1wTrend_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Williams fractals on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    bearish_fractal, bullish_fractal = compute_williams_fractals(
        df_1d['high'].values,
        df_1d['low'].values,
    )
    # Fractals need 2-bar confirmation after the center bar
    bearish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bearish_fractal, additional_delay_bars=2
    )
    bullish_fractal_aligned = align_htf_to_ltf(
        prices, df_1d, bullish_fractal, additional_delay_bars=2
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: bullish fractal breakout (price > fractal high) + bullish 1w trend
            if (close[i] > bullish_fractal_aligned[i] and 
                ema34_1w_aligned[i] > 0):  # EMA > 0 always true, but we check slope via price vs EMA
                # Additional trend filter: price above 1w EMA
                if close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: bearish fractal breakdown (price < fractal low) + bearish 1w trend
            elif (close[i] < bearish_fractal_aligned[i] and 
                  ema34_1w_aligned[i] > 0):
                # Additional trend filter: price below 1w EMA
                if close[i] < ema34_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: bearish fractal breaks OR trend turns bearish (price < 1w EMA)
            if (close[i] < bearish_fractal_aligned[i]) or (close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish fractal breaks OR trend turns bullish (price > 1w EMA)
            if (close[i] > bullish_fractal_aligned[i]) or (close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals