#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 1w EMA50 trend filter and volume confirmation.
Uses Bollinger Band width percentile to detect low volatility squeezes (BBW < 20th percentile).
Breakouts from squeeze are traded in the direction of 1w EMA50 trend.
Volume spike (>2.0x 20-period MA) confirms breakout momentum.
Designed for 6h timeframe to capture explosive moves after consolidation in both bull/bear markets.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
Uses discrete position sizing (0.25) to balance return and fee drag.
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
    
    # Calculate Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate BB width percentile rank (50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need BB percentile and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width_percentile[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Squeeze condition: BB width at or below 20th percentile
        squeeze = bb_width_percentile[i] <= 20.0
        
        # Trend filter: close > 1w EMA50 = uptrend, close < 1w EMA50 = downtrend
        trend_up = close[i] > ema_50_1w_aligned[i]
        trend_down = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Break above upper BB AND uptrend AND volume spike AND squeeze
            if close[i] > bb_upper[i] and trend_up and volume_spike[i] and squeeze:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower BB AND downtrend AND volume spike AND squeeze
            elif close[i] < bb_lower[i] and trend_down and volume_spike[i] and squeeze:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: break of opposite Bollinger Band
            exit_signal = False
            if position == 1:
                # Exit long on break below lower BB
                if close[i] < bb_lower[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above upper BB
                if close[i] > bb_upper[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Bollinger_Squeeze_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0