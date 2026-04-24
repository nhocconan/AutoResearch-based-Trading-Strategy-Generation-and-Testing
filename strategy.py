#!/usr/bin/env python3
"""
Hypothesis: 6h ADX + Elder Ray + 12h EMA trend filter.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for trend filter (price above/below EMA34).
- Entry: Long when ADX > 25 (trending) AND Elder Ray bull power > 0 AND price > 12h EMA34.
         Short when ADX > 25 (trending) AND Elder Ray bear power < 0 AND price < 12h EMA34.
- Exit: ADX < 20 (range) OR Elder Ray power contradicts position OR price crosses 12h EMA34 opposite.
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- ADX filters ranging markets where Elder Ray gives false signals.
- Elder Ray confirms trend strength behind price action.
- Works in bull markets (long when bullish aligned) and bear markets (short when bearish aligned).
- Estimated trades: ~100 total over 4 years (~25/year) based on ADX>25 frequency with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def wilder_smoothing(values, period):
    """Calculate Wilder's smoothing (used in ADX)."""
    return pd.Series(values).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h trend filter: EMA34
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    ema34_12h = ema(df_12h['close'].values, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h, additional_delay_bars=1)
    
    # ADX calculation (14-period)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = wilder_smoothing(tr, 14)
    plus_di = 100 * wilder_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilder_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smoothing(dx, 14)
    
    # Elder Ray on 6h (bull power = high - EMA13, bear power = low - EMA13)
    ema13 = ema(close, 13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need sufficient data for ADX/EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: ADX < 20 (range) OR Elder Ray contradicts OR price crosses 12h EMA34 opposite
        if position != 0:
            # Exit long: ADX < 20 OR bear power > 0 OR price falls below 12h EMA34
            if position == 1:
                if adx[i] < 20 or bear_power[i] > 0 or curr_close < ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: ADX < 20 OR bull power < 0 OR price rises above 12h EMA34
            elif position == -1:
                if adx[i] < 20 or bull_power[i] < 0 or curr_close > ema34_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: ADX > 25 (trending) AND Elder Ray aligned AND 12h trend aligned
        if position == 0:
            # Long: ADX > 25 AND bull power > 0 AND bullish 12h trend
            if adx[i] > 25 and bull_power[i] > 0 and curr_close > ema34_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 AND bear power < 0 AND bearish 12h trend
            elif adx[i] > 25 and bear_power[i] < 0 and curr_close < ema34_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_ADX_ElderRay_12hEMA34_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0