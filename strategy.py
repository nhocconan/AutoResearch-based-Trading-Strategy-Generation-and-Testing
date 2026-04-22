#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day trend filter.
Long when Bollinger Bands width < 20th percentile (squeeze) and price breaks above upper band, with 1-day EMA50 uptrend.
Short when Bollinger Bands width < 20th percentile and price breaks below lower band, with 1-day EMA50 downtrend.
Exit when price crosses the middle Bollinger Band (20-period SMA).
Bollinger squeeze indicates low volatility primed for expansion; breakout captures the move.
EMA50 filter ensures trading with the higher timeframe trend to avoid counter-trend whipsaws.
Works in bull markets by catching breakouts in uptrends and in bear markets by catching breakdowns in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_mult = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_mult * std
    lower = sma - bb_mult * std
    bb_width = upper - lower
    
    # Bollinger Band width percentile (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=1).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Load 1-day EMA50 for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):
        # Skip if data not ready
        if np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(bb_width_percentile[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bollinger squeeze (<20th percentile) + break above upper band + 1-day EMA50 uptrend
            if (bb_width_percentile[i] < 20 and 
                close[i] > upper[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bollinger squeeze (<20th percentile) + break below lower band + 1-day EMA50 downtrend
            elif (bb_width_percentile[i] < 20 and 
                  close[i] < lower[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price crosses the middle Bollinger Band (SMA)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below SMA
                if close[i] < sma[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above SMA
                if close[i] > sma[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BB_Squeeze_Breakout_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0