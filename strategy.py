#!/usr/bin/env python3
"""
Hypothesis: 4h Bollinger Band squeeze breakout with 1d EMA50 trend and volume confirmation.
Long when price closes above upper Bollinger Band AND 1d EMA50 rising AND volume > 1.5x 20-period average.
Short when price closes below lower Bollinger Band AND 1d EMA50 falling AND volume > 1.5x 20-period average.
Exit when price crosses the middle Bollinger Band (20-period SMA).
Uses Bollinger Band width percentile regime filter: only trade when BBW < 30th percentile (low volatility squeeze).
Position size: 0.25. Target: 20-50 trades/year per symbol.
Designed to capture explosive moves after low volatility periods, working in both bull and bear markets by using HTF trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Bollinger Bands (20, 2) on primary timeframe
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # Bollinger Band Width (normalized) for regime filter
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Calculate 50-period percentile rank of BB width for regime filter
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Ensure warmup for BB and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price closes above upper BB AND 1d EMA50 rising AND volume spike AND low volatility regime
            if (close[i] > bb_upper[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and  # EMA50 rising
                volume[i] > 1.5 * vol_ma_val and
                bb_width_percentile[i] < 0.30):  # BB width < 30th percentile (squeeze)
                signals[i] = 0.25
                position = 1
            # Short: price closes below lower BB AND 1d EMA50 falling AND volume spike AND low volatility regime
            elif (close[i] < bb_lower[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and  # EMA50 falling
                  volume[i] > 1.5 * vol_ma_val and
                  bb_width_percentile[i] < 0.30):  # BB width < 30th percentile (squeeze)
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses middle Bollinger Band (mean reversion)
            if position == 1 and close[i] < bb_middle[i]:
                exit_signal = True
            elif position == -1 and close[i] > bb_middle[i]:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BB_Squeeze_EMA50_Volume"
timeframe = "4h"
leverage = 1.0