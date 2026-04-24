#!/usr/bin/env python3
"""
Hypothesis: 1h Bollinger Band Squeeze with 4h/1d regime filter and volume confirmation.
- Long when price breaks above upper BB(20,2) AND 4h close > 1d EMA50 (bullish bias) AND volume > 2x 20-period average
- Short when price breaks below lower BB(20,2) AND 4h close < 1d EMA50 (bearish bias) AND volume > 2x 20-period average
- Exit when price returns to middle BB(20) OR volatility expands (BB width > 50th percentile of last 50)
- Uses Bollinger Squeeze (low volatility breakout) for entry timing, HTF for directional bias, volume for confirmation
- Designed to work in both bull (breakouts with trend) and bear (breakdowns with trend) markets
- Signal size: 0.20 discrete levels to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year)
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
    
    # Calculate Bollinger Bands (20,2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = basis + 2 * dev
    lower_bb = basis - 2 * dev
    
    # Calculate BB width for squeeze detection
    bb_width = (upper_bb - lower_bb) / basis
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    
    # Calculate 4h EMA20 for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d EMA50 for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = close_s.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2 * vol_ma)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 20  # Need BB, EMAs, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(basis[i]) or np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper BB AND 4h EMA20 > 1d EMA50 (bullish bias) AND volume confirmation
            if close[i] > upper_bb[i] and ema_20_4h_aligned[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower BB AND 4h EMA20 < 1d EMA50 (bearish bias) AND volume confirmation
            elif close[i] < lower_bb[i] and ema_20_4h_aligned[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to middle BB OR volatility expands (BB width > 50th percentile)
            if close[i] < basis[i] or bb_width_percentile[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to middle BB OR volatility expands
            if close[i] > basis[i] or bb_width_percentile[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_BBandSqueeze_4hEMA20_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0