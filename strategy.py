#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1w Exponential Moving Average for trend and 1d Bollinger Bands for mean reversion.
# Long when price crosses above 1w EMA with Bollinger Band squeeze (low volatility) and volume confirmation.
# Short when price crosses below 1w EMA with Bollinger Band squeeze and volume confirmation.
# Uses Bollinger Band width percentile to identify low volatility regimes for breakout entries.
# Designed for low trade frequency (15-25/year) to avoid fee drag. Combines trend following with volatility-based timing.

name = "6h_1wEMA_BB_Squeeze_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA (21 period on weekly data)
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d Bollinger Bands (20, 2.0)
    close_1d = df_1d['close'].values
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20_1d + (std_20_1d * 2.0)
    lower_bb = sma_20_1d - (std_20_1d * 2.0)
    bb_width = upper_bb - lower_bb
    
    # Calculate Bollinger Band width percentile (252 trading days ~ 1 year)
    bb_width_pct = pd.Series(bb_width).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align 1w and 1d indicators to 6h timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    
    # Volume confirmation: 6h volume > 1.5x 50-period EMA
    vol_ema = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(bb_width_pct_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price crosses above 1w EMA + BB squeeze (width < 20th percentile) + volume
            if (close[i] > ema_21_1w_aligned[i] and 
                bb_width_pct_aligned[i] < 20.0 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below 1w EMA + BB squeeze (width < 20th percentile) + volume
            elif (close[i] < ema_21_1w_aligned[i] and 
                  bb_width_pct_aligned[i] < 20.0 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1w EMA
            if close[i] < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1w EMA
            if close[i] > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals