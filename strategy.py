#!/usr/bin/env python3
"""
6h_RSI_Aroon_Trend_Momentum
Hypothesis: Combines RSI momentum with Aroon trend strength on 6h timeframe, 
using 12h trend as filter. Aroon identifies trend strength (0-100 scale), 
RSI identifies overbought/oversold conditions within the trend. 
In bull markets: buy when Aroon-up > 70 and RSI < 40 (pullback in uptrend)
In bear markets: sell when Aroon-down > 70 and RSI > 60 (bounce in downtrend)
Volatility filter ensures trades only in sufficient momentum environments.
Designed to work in both trending and ranging markets with controlled frequency.
"""

name = "6h_RSI_Aroon_Trend_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtr_data import get_htf_data, align_htf_to_ltf

def calculate_aroon(high, low, period=25):
    """Calculate Aroon Up and Aroon Down indicators"""
    n = len(high)
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(period, n):
        # Look back period periods
        period_high = high[i-period+1:i+1]
        period_low = low[i-period+1:i+1]
        
        # Find periods since highest high and lowest low
        high_idx = np.argmax(period_high)
        low_idx = np.argmin(period_low)
        
        aroon_up[i] = ((period - 1 - high_idx) / (period - 1)) * 100
        aroon_down[i] = ((period - 1 - low_idx) / (period - 1)) * 100
    
    return aroon_up, aroon_down

def calculate_rsi(close, period=14):
    """Calculate RSI with Wilder's smoothing"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha = 1/period)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    for i in range(period+1, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Aroon on 6h
    aroon_up, aroon_down = calculate_aroon(high, low, period=25)
    
    # Calculate RSI on 6h
    rsi = calculate_rsi(close, period=14)
    
    # Align 12h EMA50 trend to 6h timeframe (trend = close > EMA50)
    trend_12h = (close_12h > ema_50_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or 
            np.isnan(rsi[i]) or np.isnan(trend_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h uptrend AND Aroon-up > 70 (strong uptrend) AND RSI < 40 (pullback)
            if (trend_12h_aligned[i] == 1 and aroon_up[i] > 70 and rsi[i] < 40 and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend AND Aroon-down > 70 (strong downtrend) AND RSI > 60 (bounce)
            elif (trend_12h_aligned[i] == 0 and aroon_down[i] > 70 and rsi[i] > 60 and volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 12h trend turns down OR Aroon-down > 60 (weakening uptrend)
            if trend_12h_aligned[i] == 0 or aroon_down[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: 12h trend turns up OR Aroon-up > 60 (weakening downtrend)
            if trend_12h_aligned[i] == 1 or aroon_up[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals