#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Bollinger Band squeeze breakout with volume confirmation and daily trend filter.
# In low volatility (Bollinger Band width < 20th percentile), price is primed for breakout.
# Long when price breaks above upper BB with volume > 1.5x 20-period average and daily trend up.
# Short when price breaks below lower BB with volume confirmation and daily trend down.
# Uses 1-day trend filter to avoid counter-trend trades. Designed for low trade frequency (15-25/year).

name = "12h_BBSqueeze_Breakout_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) - calculate on close
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (20-period lookback) - identifies squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Daily trend filter - use 1-day EMA crossover
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_fast_1d = pd.Series(close_1d).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    daily_trend_up = ema_fast_1d > ema_slow_1d  # True for uptrend
    
    # Align daily trend to 12h timeframe
    daily_trend_up_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for BB calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_width_percentile[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(daily_trend_up_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Bollinger Band squeeze condition: width < 20th percentile
            bb_squeeze = bb_width_percentile[i] < 20
            
            if bb_squeeze:
                # Long setup: break above upper BB with volume and daily uptrend
                if (close[i] > bb_upper[i] and 
                    daily_trend_up_aligned[i] > 0.5 and 
                    vol_confirm[i]):
                    signals[i] = 0.25
                    position = 1
                # Short setup: break below lower BB with volume and daily downtrend
                elif (close[i] < bb_lower[i] and 
                      daily_trend_up_aligned[i] <= 0.5 and 
                      vol_confirm[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price returns to middle BB or trend turns down
            if close[i] < bb_middle[i] or daily_trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to middle BB or trend turns up
            if close[i] > bb_middle[i] or daily_trend_up_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals