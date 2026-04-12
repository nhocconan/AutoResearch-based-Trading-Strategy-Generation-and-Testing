#!/usr/bin/env python3
"""
6h_1w_1d_engulfing_reversal
Hypothesis: 6-hour strategy combining weekly trend filter with daily bullish/bearish engulfing patterns.
Enters long when daily bullish engulfing forms above weekly EMA50 uptrend; short when daily bearish engulfing forms below weekly EMA50 downtrend.
Uses volume confirmation to avoid false signals. Targets reversals in trending markets.
Designed to work in both bull and bear markets by following weekly trend while capturing daily reversals.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful reversals.
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
    
    # Get daily data for engulfing patterns
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily bullish and bearish engulfing patterns
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Bullish engulfing: current candle bullish and engulfs previous bearish candle
    bullish_engulf = (
        (close_1d > open_1d) &  # current bullish
        (open_1d < close_1d) &  # redundant but clear
        (open_1d <= close_1d) &  # current bullish
        (open_1d >= close_1d) &  # this line is wrong, let me fix
        (close_1d > open_1d) &  # current bullish
        (open_1d < close_1d) &  # previous bearish condition
        (close_1d[-1] < open_1d[-1])  # this approach won't work vectorized
    )
    # Let me implement properly
    bullish_engulf = (
        (close_1d > open_1d) &  # current candle bullish
        (open_1d.shift(1) > close_1d.shift(1)) &  # previous candle bearish
        (close_1d > open_1d.shift(1)) &  # current close > previous open
        (open_1d < close_1d.shift(1))    # current open < previous close
    )
    # Handle shift for first element
    bullish_engulf = np.where(np.arange(len(bullish_engulf)) > 0, 
                             bullish_engulf.values if hasattr(bullish_engulf, 'values') else bullish_engulf, False)
    bullish_engulf = np.concatenate([[False], bullish_engulf[:-1]]) if len(bullish_engulf) > 0 else np.array([])
    
    # Bearish engulfing: current candle bearish and engulfs previous bullish candle
    bearish_engulf = (
        (close_1d < open_1d) &  # current candle bearish
        (open_1d.shift(1) < close_1d.shift(1)) &  # previous candle bullish
        (close_1d < open_1d.shift(1)) &  # current close < previous open
        (open_1d > close_1d.shift(1))    # current open > previous close
    )
    bearish_engulf = np.concatenate([[False], bearish_engulf[:-1]]) if len(bearish_engulf) > 0 else np.array([])
    
    # Reimplement engulfing properly using loops for clarity and correctness
    bullish_engulf = np.zeros(len(close_1d), dtype=bool)
    bearish_engulf = np.zeros(len(close_1d), dtype=bool)
    
    for i in range(1, len(close_1d)):
        # Bullish engulfing: current bullish, previous bearish, and engulfs
        if (close_1d[i] > open_1d[i] and  # current bullish
            open_1d[i-1] > close_1d[i-1] and  # previous bearish
            close_1d[i] > open_1d[i-1] and  # current close > previous open
            open_1d[i] < close_1d[i-1]):    # current open < previous close
            bullish_engulf[i] = True
            
        # Bearish engulfing: current bearish, previous bullish, and engulfs
        if (close_1d[i] < open_1d[i] and  # current bearish
            open_1d[i-1] < close_1d[i-1] and  # previous bullish
            close_1d[i] < open_1d[i-1] and  # current close < previous open
            open_1d[i] > close_1d[i-1]):    # current open > previous close
            bearish_engulf[i] = True
    
    # Align engulfing signals to 6h timeframe
    bullish_engulf_6h = align_htf_to_ltf(prices, df_1d, bullish_engulf.astype(float))
    bearish_engulf_6h = align_htf_to_ltf(prices, df_1d, bearish_engulf.astype(float))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend direction
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate ATR for volatility filter and position sizing
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bullish_engulf_6h[i]) or np.isnan(bearish_engulf_6h[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[max(0, i-20):i])
            volume_filter = volume[i] > vol_ma * 1.3
        else:
            volume_filter = False
        
        # Trend filter from weekly EMA50
        uptrend_1w = close[i] > ema50_1w_aligned[i]
        downtrend_1w = close[i] < ema50_1w_aligned[i]
        
        # Fixed position size
        position_size = 0.25
        
        # Entry conditions: Engulfing pattern with volume and weekly trend confirmation
        long_entry = bullish_engulf_6h[i] > 0.5 and volume_filter and uptrend_1w
        short_entry = bearish_engulf_6h[i] > 0.5 and volume_filter and downtrend_1w
        
        # Exit conditions: opposite engulfing or trend change
        long_exit = bearish_engulf_6h[i] > 0.5 or not uptrend_1w
        short_exit = bullish_engulf_6h[i] > 0.5 or not downtrend_1w
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_engulfing_reversal"
timeframe = "6h"
leverage = 1.0