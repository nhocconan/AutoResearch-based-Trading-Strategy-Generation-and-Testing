#!/usr/bin/env python3
"""
6H ADX + Bollinger Band Squeeze Reversal Strategy
Long when Bollinger Bands squeeze (BBW at 20-period low) + ADX < 20 (range) + price touches lower band
Short when Bollinger Bands squeeze + ADX < 20 + price touches upper band
Exit when price crosses middle Bollinger band
Uses volatility contraction to identify range exhaustion points for mean reversion.
Works in both bull and bear markets by fading extremes during low volatility periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_bb_squeeze_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Bollinger Bands (20, 2) ===
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_middle = sma20
    
    # === Bollinger Band Width (for squeeze detection) ===
    bb_width = (bb_upper - bb_lower) / (bb_middle + 1e-10)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: np.percentile(x, 20) if len(x) == 50 else np.nan, raw=True
    ).values
    squeeze_condition = bb_width <= bb_width_percentile  # BBW at 20th percentile or lower
    
    # === ADX (14) for trend strength ===
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the DM and TR values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Calculate Directional Indicators
    plus_di = 100 * plus_dm_sum / (tr_sum + 1e-10)
    minus_di = 100 * minus_dm_sum / (tr_sum + 1e-10)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Range condition: ADX < 20 indicates ranging market
    range_condition = adx < 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_middle[i]) or
            np.isnan(squeeze_condition[i]) or np.isnan(range_condition[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above middle Bollinger band
            if close[i] > bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below middle Bollinger band
            if close[i] < bb_middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Entry conditions: Bollinger squeeze + ranging market + price at band extreme
            if squeeze_condition[i] and range_condition[i]:
                if close[i] <= bb_lower[i]:
                    # Price at or below lower band -> long (mean reversion up)
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= bb_upper[i]:
                    # Price at or above upper band -> short (mean reversion down)
                    position = -1
                    signals[i] = -0.25
    
    return signals