#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily ATR-based breakout with volume confirmation and trend filter
# Daily ATR breakout captures volatility expansion in both bull and bear markets
# Breakout above/below previous day's close ± 1.5x ATR with volume > 1.5x 20-period average
# Trend filter: 50-period EMA on 12h to avoid counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_ATRBreakout_VolumeTrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily ATR and close for breakout levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for ATR calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # True Range calculation
    tr1 = prev_high - prev_low
    tr2 = np.abs(prev_high - prev_close)
    tr3 = np.abs(prev_low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) calculation
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Breakout levels: previous close ± 1.5 * ATR
    upper_break = prev_close + (1.5 * atr_14)
    lower_break = prev_close - (1.5 * atr_14)
    
    # Align daily levels to 12h timeframe
    upper_break_aligned = align_htf_to_ltf(prices, df_1d, upper_break)
    lower_break_aligned = align_htf_to_ltf(prices, df_1d, lower_break)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 50-period EMA on 12h timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_break_aligned[i]) or np.isnan(lower_break_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper level with volume confirmation and uptrend
            if close[i] > upper_break_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower level with volume confirmation and downtrend
            elif close[i] < lower_break_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below previous day's close (failed breakout) or reaches 2x ATR target
            if close[i] < prev_close_aligned[i] or close[i] > upper_break_aligned[i] + (atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above previous day's close (failed breakdown) or reaches 2x ATR target
            if close[i] > prev_close_aligned[i] or close[i] < lower_break_aligned[i] - (atr_14_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Pre-calculate aligned arrays for prev_close and atr_14 to avoid recomputation in loop
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)