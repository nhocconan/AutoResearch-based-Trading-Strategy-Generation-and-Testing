#!/usr/bin/env python3
# 12H_Price_Action_Reversal_V1
# Hypothesis: Combines 12h price action reversals with 1d volume confirmation and 1d trend filter.
# Uses rejection at daily support/resistance (engulfing patterns) to capture mean reversion in range-bound markets
# and breakouts in trending markets. Designed for 12h timeframe with low trade frequency (15-25/year) to minimize
# fee drag. Works in both bull and bear regimes by adapting to price action context rather than fixed indicators.
# Target: 60-100 total trades over 4 years (15-25/year per symbol).

name = "12H_Price_Action_Reversal_V1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Price action signals: bullish/bearish engulfing patterns
    bullish_engulf = (close > open_price) & (open_price > np.roll(close, 1)) & (close > np.roll(high, 1))
    bearish_engulf = (close < open_price) & (open_price < np.roll(close, 1)) & (close < np.roll(low, 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure we have EMA and volume data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i]) or vol_avg_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 12h volume > 1.5x average 1d volume
        volume_filter = volume[i] > 1.5 * vol_avg_aligned[i]
        
        if position == 0:
            # Long: Bullish engulfing + above 1d EMA50 + volume confirmation
            if (bullish_engulf[i] and 
                close[i] > ema50_aligned[i] and
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing + below 1d EMA50 + volume confirmation
            elif (bearish_engulf[i] and 
                  close[i] < ema50_aligned[i] and
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Opposite engulfing pattern or price crosses 1d EMA50
            exit_signal = False
            if position == 1:
                exit_signal = bearish_engulf[i] or close[i] < ema50_aligned[i]
            else:  # position == -1
                exit_signal = bullish_engulf[i] or close[i] > ema50_aligned[i]
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals