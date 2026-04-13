#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout with daily trend filter + volume confirmation
    # Long: BB squeeze (BW < 20th percentile) AND price > upper BB AND daily EMA50 > EMA200 AND volume > 1.5x avg
    # Short: BB squeeze (BW < 20th percentile) AND price < lower BB AND daily EMA50 < EMA200 AND volume > 1.5x avg
    # Exit: price crosses middle BB (20-period SMA) OR opposite BB touch
    # Using 6h timeframe for optimal trade frequency (target 12-37/year), Bollinger squeeze to identify low volatility
    # periods before expansion, daily EMA crossover for trend filter, and volume confirmation to avoid false breakouts.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 and EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMAs to 6h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = np.full(n, np.nan)
    bb_ma = np.full(n, np.nan)
    
    for i in range(bb_period, n):
        bb_ma[i] = np.mean(close[i-bb_period:i])
        bb_std[i] = np.std(close[i-bb_period:i])
    
    bb_upper = bb_ma + (2 * bb_std)
    bb_lower = bb_ma - (2 * bb_std)
    bb_middle = bb_ma
    
    # Calculate Bollinger Band Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate percentile rank of BB width (20-period lookback)
    bb_width_percentile = np.full(n, np.nan)
    lookback = 20
    
    for i in range(lookback, n):
        window = bb_width[i-lookback:i]
        current = bb_width[i]
        if not np.isnan(current) and not np.all(np.isnan(window)):
            # Calculate percentile: percentage of values in window <= current
            valid_window = window[~np.isnan(window)]
            if len(valid_window) > 0:
                bb_width_percentile[i] = (np.sum(valid_window <= current) / len(valid_window)) * 100
    
    # Bollinger squeeze condition: width < 20th percentile
    bb_squeeze = bb_width_percentile < 20
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(bb_squeeze[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter conditions
        bullish_trend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        bearish_trend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Bollinger Band conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        bb_middle_cross_up = (close[i] > bb_middle[i]) and (prices['close'].iloc[i-1] <= bb_middle[i-1]) if i > 0 else False
        bb_middle_cross_down = (close[i] < bb_middle[i]) and (prices['close'].iloc[i-1] >= bb_middle[i-1]) if i > 0 else False
        
        # Entry logic: Squeeze breakout + trend alignment + volume confirmation
        long_entry = bb_squeeze[i] and bb_breakout_up and bullish_trend and volume_spike[i]
        short_entry = bb_squeeze[i] and bb_breakout_down and bearish_trend and volume_spike[i]
        
        # Exit logic: middle BB cross or opposite BB touch
        long_exit = bb_middle_cross_down or bb_breakout_down
        short_exit = bb_middle_cross_up or bb_breakout_up
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_bb_squeeze_trend_volume_v1"
timeframe = "6h"
leverage = 1.0