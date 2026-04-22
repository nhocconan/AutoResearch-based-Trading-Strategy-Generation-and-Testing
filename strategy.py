#!/usr/bin/env python3
"""
Hypothesis: 4-hour Bollinger Band squeeze breakout with 1-day trend filter and volume confirmation.
Long when price breaks above upper BB during low volatility (BBW < 50th percentile) with 1-day EMA50 up and volume spike.
Short when price breaks below lower BB during low volatility with 1-day EMA50 down and volume squeeze.
Exit when price re-enters the Bollinger Bands.
Uses volatility contraction/expansion to capture breakouts with low false signals.
Works in both bull and bear markets by following daily trend direction.
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
    volume = prices['volume'].values
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Bollinger Band Width for squeeze detection
    bb_width = (bb_upper - bb_lower) / bb_middle
    # Percentile lookback for squeeze definition (50th percentile = median)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=20).rank(pct=True).values
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_percentile[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility squeeze condition (low volatility environment)
        volatility_squeeze = bb_width_percentile[i] < 0.5  # Below 50th percentile
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_30[i]
        
        if position == 0:
            # Long: Price breaks above upper BB during low volatility with 1-day EMA50 up and volume spike
            if (close[i] > bb_upper[i] and 
                volatility_squeeze and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB during low volatility with 1-day EMA50 down and volume spike
            elif (close[i] < bb_lower[i] and 
                  volatility_squeeze and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price re-enters the Bollinger Bands
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below upper BB (or re-enters bands)
                if close[i] < bb_middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above lower BB (or re-enters bands)
                if close[i] > bb_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_BollingerSqueeze_Breakout_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0