#!/usr/bin/env python3
name = "4h_Adaptive_Kelly_RSI2_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import sqrt

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data once for trend filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4-hour RSI(2) for mean reversion signal
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to RMA)
    alpha = 1.0 / 2
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    
    for i in range(2, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands width for volatility regime filter
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std_dev
    lower = sma - bb_std * std_dev
    bb_width = (upper - lower) / sma
    
    # Volume filter: above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, bb_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_4h[i]) or np.isnan(rsi[i]) or 
            np.isnan(bb_width[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma[i] * 1.5
        bb_condition = bb_width[i] < 0.05  # Low volatility (squeeze)
        trend_up = ema_50_4h[i] > ema_50_4h[i-1]
        trend_down = ema_50_4h[i] < ema_50_4h[i-1]
        
        # Adaptive Kelly sizing based on RSI extremity
        if rsi[i] < 10:  # Deep overshoot
            kelly_fraction = 0.35
        elif rsi[i] < 20:  # Oversold
            kelly_fraction = 0.25
        elif rsi[i] > 90:  # Deep overbought
            kelly_fraction = -0.35
        elif rsi[i] > 80:  # Overbought
            kelly_fraction = -0.25
        else:
            kelly_fraction = 0.0
        
        if position == 0:
            # Enter long in uptrend on RSI oversold with volume and low volatility
            if kelly_fraction > 0 and trend_up and vol_condition and bb_condition:
                signals[i] = kelly_fraction
                position = 1
            # Enter short in downtrend on RSI overbought with volume and low volatility
            elif kelly_fraction < 0 and trend_down and vol_condition and bb_condition:
                signals[i] = kelly_fraction
                position = -1
        elif position == 1:
            # Exit: RSI returns to neutral or trend breaks
            if rsi[i] > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = kelly_fraction
        elif position == -1:
            # Exit: RSI returns to neutral or trend breaks
            if rsi[i] < 50 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = kelly_fraction
    
    return signals

# Hypothesis: Adaptive Kelly RSI(2) mean reversion with 1d trend filter
# - RSI(2) captures short-term mean reversion extremes (<10 oversold, >90 overbought)
# - Kelly sizing scales position based on signal strength (0.25 for moderate, 0.35 for extreme)
# - 1-day EMA50 ensures trades align with higher-timeframe trend
# - Volume confirmation (>1.5x average) and Bollinger Band squeeze (<5% width) filter low-quality signals
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Adaptive sizing reduces risk during weak signals, increases during strong setups
# - Target: 20-40 trades/year to minimize fee drag while capturing high-probability mean reversion
# - Uses 1d timeframe for trend, 4h for execution - avoids look-ahead with proper alignment