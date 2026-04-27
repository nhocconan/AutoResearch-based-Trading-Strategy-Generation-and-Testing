#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d multi-timeframe confirmation.
# Uses 4h ADX for trend strength filter and 1d Bollinger Bands for mean reversion zones.
# Long when price touches lower BB in 1d, 4h ADX > 25 (trending), and closes above 1h EMA20.
# Short when price touches upper BB in 1d, 4h ADX > 25, and closes below 1h EMA20.
# Exit on opposite BB touch or when ADX weakens (< 20).
# Session filter: 08-20 UTC to avoid low-volume Asian session.
# Position size: 0.20 (discrete to minimize fee churn).
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for ADX trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX(14) on 4h
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align to original index
        
        # Directional Movement
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value: simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / atr
        minus_di = 100 * wilder_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_4h = calculate_adx(high_4h, low_4h, close_4h, 14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Get 1d data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands(20,2) on 1d
    def calculate_bb(close, period=20, std_dev=2):
        sma = np.full_like(close, np.nan)
        std = np.full_like(close, np.nan)
        for i in range(len(close)):
            if i >= period - 1:
                sma[i] = np.mean(close[i-period+1:i+1])
                std[i] = np.std(close[i-period+1:i+1])
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return upper, lower
    
    bb_upper_1d, bb_lower_1d = calculate_bb(close_1d, 20, 2)
    bb_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_1d)
    bb_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_1d)
    
    # Calculate 1h EMA20 for entry timing
    close_series = pd.Series(close)
    ema20_1h = close_series.ewm(span=20, adjust=False, min_periods=20).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(bb_upper_1d_aligned[i]) or 
            np.isnan(bb_lower_1d_aligned[i]) or np.isnan(ema20_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long condition: price touches lower BB (mean reversion), 4h ADX > 25 (trending), close above EMA20
        if (low[i] <= bb_lower_1d_aligned[i] and 
            adx_4h_aligned[i] > 25 and 
            close[i] > ema20_1h[i]):
            signals[i] = 0.20
            position = 1
        # Short condition: price touches upper BB, 4h ADX > 25, close below EMA20
        elif (high[i] >= bb_upper_1d_aligned[i] and 
              adx_4h_aligned[i] > 25 and 
              close[i] < ema20_1h[i]):
            signals[i] = -0.20
            position = -1
        # Exit conditions: price touches opposite BB or ADX weakens (< 20)
        elif position == 1 and (high[i] >= bb_upper_1d_aligned[i] or adx_4h_aligned[i] < 20):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (low[i] <= bb_lower_1d_aligned[i] or adx_4h_aligned[i] < 20):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_BB_Touch_ADXTrendFilter_EMA20"
timeframe = "1h"
leverage = 1.0