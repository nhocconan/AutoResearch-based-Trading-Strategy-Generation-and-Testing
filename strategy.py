#!/usr/bin/env python3
"""
1h Bollinger Bands squeeze with 4h momentum confirmation and session filter
Hypothesis: Bollinger Bands squeeze (low volatility) precedes explosive moves. 
Only trade during high-liquidity hours (08-20 UTC) and in direction of 4h momentum (ROC > 0 long, ROC < 0 short).
Volume confirms breakout strength. Works in bull/bear by following momentum direction.
Target: 80-120 total trades over 4 years (20-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_bb_squeeze_4h_momentum_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    
    if n >= bb_period:
        for i in range(bb_period - 1, n):
            sma[i] = np.mean(close[i - bb_period + 1:i + 1])
            std = np.std(close[i - bb_period + 1:i + 1])
            bb_upper[i] = sma[i] + bb_std * std
            bb_lower[i] = sma[i] - bb_std * std
    
    # Bollinger Band Width (normalized)
    bb_width = np.full(n, np.nan)
    for i in range(bb_period - 1, n):
        if sma[i] != 0:
            bb_width[i] = (bb_upper[i] - bb_lower[i]) / sma[i]
    
    # Bollinger Squeeze: BB width below 20-period mean
    bb_width_ma = np.full(n, np.nan)
    if n >= bb_period:
        for i in range(bb_period - 1, n):
            start_idx = max(bb_period - 1, i - bb_period + 1)
            bb_width_ma[i] = np.mean(bb_width[start_idx:i + 1])
    
    squeeze = np.full(n, False)
    if n >= 2 * bb_period - 1:
        for i in range(bb_period - 1, n):
            if not np.isnan(bb_width[i]) and not np.isnan(bb_width_ma[i]):
                squeeze[i] = bb_width[i] < bb_width_ma[i]
    
    # Get 4h data for momentum (ROC)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # ROC (10-period) on 4h close
    roc_period = 10
    roc_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= roc_period + 1:
        for i in range(roc_period, len(close_4h)):
            if close_4h[i - roc_period] != 0:
                roc_4h[i] = (close_4h[i] - close_4h[i - roc_period]) / close_4h[i - roc_period]
    
    # 4h momentum direction: ROC > 0 = bullish, ROC < 0 = bearish
    momentum_4h = np.zeros(len(close_4h))
    momentum_4h[roc_4h > 0] = 1
    momentum_4h[roc_4h < 0] = -1
    
    # Align 4h momentum to 1h timeframe
    momentum_4h_aligned = align_htf_to_ltf(prices, df_4h, momentum_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on daily
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        for i in range(19, len(volume_1d)):
            vol_ma_1d[i] = np.mean(volume_1d[i - 19:i + 1])
    
    # Align volume MA to 1h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume breakout: current 1h volume > 1.5x scaled daily average
    # Scale daily volume to 1h: approx 1/24 of daily volume
    vol_threshold = vol_ma_1d_aligned / 24.0 * 1.5
    volume_breakout = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    # Start from warmup period
    start = max(bb_period, 20)  # Need enough data for BB and indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(sma[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(momentum_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price touches middle band OR momentum turns bearish
            if (close[i] <= sma[i] or momentum_4h_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.20
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price touches middle band OR momentum turns bullish
            if (close[i] >= sma[i] or momentum_4h_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.20
            bars_since_exit += 1
        else:
            # Look for entries - only during session and with minimum bars since exit
            if session_mask[i] and bars_since_exit >= 6:  # at least 6 bars between trades
                # Squeeze breakout conditions
                bull_breakout = close[i] > bb_upper[i]
                bear_breakout = close[i] < bb_lower[i]
                
                # Long: bullish breakout during squeeze with bullish 4h momentum + volume
                if squeeze[i] and bull_breakout and momentum_4h_aligned[i] == 1 and volume_breakout[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                    bars_since_exit = 0
                # Short: bearish breakout during squeeze with bearish 4h momentum + volume
                elif squeeze[i] and bear_breakout and momentum_4h_aligned[i] == -1 and volume_breakout[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                    bars_since_exit = 0
                else:
                    signals[i] = 0.0
                    bars_since_exit += 1
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals