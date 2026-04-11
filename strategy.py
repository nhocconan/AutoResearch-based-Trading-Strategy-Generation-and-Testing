#!/usr/bin/env python3
"""
6h_1d_heikinashi_engulfing_v1
Strategy: Heikin-Ashi engulfing pattern on 6h with 1d trend filter
Timeframe: 6h
Leverage: 1.0
Hypothesis: Heikin-Ashi smooths price action to filter noise. Bullish engulfing (green candle engulfing prior red) signals strength in uptrend; bearish engulfing (red candle engulfing prior green) signals weakness in downtrend. Uses 1d EMA50 to filter trend direction. Works in bull/bear by following higher timeframe trend. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_heikinashi_engulfing_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Heikin-Ashi calculation
    ha_close = (open_price + high + low + close) / 4.0
    ha_open = np.zeros_like(ha_close)
    ha_open[0] = (open_price[0] + close[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
    ha_high = np.maximum.reduce([high, ha_open, ha_close])
    ha_low = np.minimum.reduce([low, ha_open, ha_close])
    
    # Bullish engulfing: current HA green (close >= open) engulfs prior HA red (close < open)
    ha_bull_engulf = (ha_close >= ha_open) & (ha_close >= ha_open[1:]) & (ha_open <= ha_close[1:]) & (ha_close[1:] < ha_open[1:])
    # Bearish engulfing: current HA red (close < open) engulfs prior HA green (close >= open)
    ha_bear_engulf = (ha_close < ha_open) & (ha_close <= ha_open[1:]) & (ha_open >= ha_close[1:]) & (ha_close[1:] >= ha_open[1:])
    # Shift to align with current bar (pattern completes at current bar)
    ha_bull_engulf = np.roll(ha_bull_engulf, 1)
    ha_bear_engulf = np.roll(ha_bear_engulf, 1)
    ha_bull_engulf[0] = False
    ha_bear_engulf[0] = False
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if EMA data is invalid
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Entry conditions
        long_signal = ha_bull_engulf[i] and uptrend_1d
        short_signal = ha_bear_engulf[i] and downtrend_1d
        
        # Exit on opposite engulfing signal
        exit_long = position == 1 and ha_bear_engulf[i]
        exit_short = position == -1 and ha_bull_engulf[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Heikin-Ashi smooths price action to filter noise. Bullish engulfing (green candle engulfing prior red) signals strength in uptrend; bearish engulfing (red candle engulfing prior green) signals weakness in downtrend. Uses 1d EMA50 to filter trend direction. Works in bull/bear by following higher timeframe trend. Target: 50-150 total trades over 4 years.