#!/usr/bin/env python3
# 12h_engulfing_1w1d_volume_v1
# Hypothesis: Weekly and daily bullish/bearish engulfing candles combined with volume confirmation on 12h chart.
# Long when daily candle is bullish engulfing, weekly trend is up, and 12h volume > 1.5x average.
# Short when daily candle is bearish engulfing, weekly trend is down, and 12h volume > 1.5x average.
# Exit on opposite engulfing signal or when price reaches opposite engulfing level.
# Designed to capture momentum shifts at key turning points with volume confirmation.
# Target: 15-25 trades/year to minimize fee drift while capturing high-probability moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_engulfing_1w1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for engulfing patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily OHLC
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bullish engulfing: current close > previous open AND current open < previous close
    bullish_engulf = (close_1d > open_1d[:-1]) & (open_1d < close_1d[:-1])
    bullish_engulf = np.concatenate([np.array([False]), bullish_engulf])
    
    # Bearish engulfing: current close < previous open AND current open > previous close
    bearish_engulf = (close_1d < open_1d[:-1]) & (open_1d > close_1d[:-1])
    bearish_engulf = np.concatenate([np.array([False]), bearish_engulf])
    
    # Align daily engulfing signals to 12h chart
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bullish_engulf.astype(float))
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1d, bearish_engulf.astype(float))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly trend: based on close vs open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_up = close_1w > open_1w
    weekly_down = close_1w < open_1w
    
    # Align weekly trend to 12h chart
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up.astype(float))
    weekly_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_down.astype(float))
    
    # Volume confirmation: 24-period average (2 days of 12h data)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(bullish_engulf_aligned[i]) or np.isnan(bearish_engulf_aligned[i]) or \
           np.isnan(weekly_up_aligned[i]) or np.isnan(weekly_down_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish engulfing signal or opposite conditions
            if bearish_engulf_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: bullish engulfing signal or opposite conditions
            if bullish_engulf_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: daily bullish engulfing, weekly up trend, volume confirmation
            if bullish_engulf_aligned[i] > 0.5 and weekly_up_aligned[i] > 0.5 and volume_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: daily bearish engulfing, weekly down trend, volume confirmation
            elif bearish_engulf_aligned[i] > 0.5 and weekly_down_aligned[i] > 0.5 and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals