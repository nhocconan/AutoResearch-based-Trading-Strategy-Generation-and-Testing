#!/usr/bin/env python3
"""
Hypothesis: 6h Bollinger Band squeeze breakout with 12h EMA trend filter and 1d volume confirmation.
- Primary timeframe: 6h for execution.
- Bollinger Band Width (BBW) percentile identifies low volatility squeeze (BBW < 20th percentile).
- Breakout direction determined by 12h EMA50 trend (price > EMA50 = bullish, < EMA50 = bearish).
- Volume confirmation: current 6h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- In ranging markets (no squeeze): mean reversion at Bollinger Bands (touch upper/lower band + reversal).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in both bull and bear markets: squeeze breakouts capture expansion phases, mean reversion works in consolidation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 6h
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + (2.0 * dev)
    lower_band = basis - (2.0 * dev)
    bb_width = ((upper_band - lower_band) / basis) * 100  # BBW as percentage
    
    # BBW percentile lookback (50 periods ~ 6.5 days on 6h)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50.0, raw=False
    ).values
    squeeze = bb_width_percentile < 20  # BBW in bottom 20% = squeeze
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    uptrend_12h = close > ema_50_12h_aligned  # 6h price above 12h EMA50
    downtrend_12h = close < ema_50_12h_aligned  # 6h price below 12h EMA50
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # We'll use 6h volume MA for confirmation (more responsive)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough for BBW percentile and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if squeeze[i]:  # Squeeze breakout: trade in direction of 12h trend
                    if uptrend_12h[i]:  # Bullish breakout
                        signals[i] = 0.25
                        position = 1
                    elif downtrend_12h[i]:  # Bearish breakout
                        signals[i] = -0.25
                        position = -1
                else:  # No squeeze: mean reversion at Bollinger Bands
                    # Long when price touches lower band and shows reversal (close > low)
                    if curr_low <= lower_band[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper band and shows reversal (close < high)
                    elif curr_high >= upper_band[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below basis OR squeeze breaks in opposite direction
            if curr_close < basis[i] or (squeeze[i] and downtrend_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above basis OR squeeze breaks in opposite direction
            if curr_close > basis[i] or (squeeze[i] and uptrend_12h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BB_Squeeze_Breakout_12hEMA50_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0