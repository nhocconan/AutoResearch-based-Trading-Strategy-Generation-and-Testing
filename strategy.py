#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout + 12h ADX trend filter
    # Only trade breakouts when volatility is low (BB width < 20th percentile) and 12h ADX > 25
    # Direction: breakout above upper band = long, below lower band = short
    # Uses discrete sizing 0.25 to minimize fee churn. Target: 15-35 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = (upper - lower) / sma * 100  # percentage
    
    # Calculate 6h BB width percentile (lookback 50 periods for regime)
    def calculate_percentile(arr, lookback=50):
        n = len(arr)
        percentile = np.full(n, np.nan)
        for i in range(lookback, n):
            window = arr[i-lookback:i]
            if not np.all(np.isnan(window)):
                percentile[i] = np.percentile(window[~np.isnan(window)], 20)  # 20th percentile
        return percentile
    
    bb_width_20th = calculate_percentile(bb_width, 50)
    squeeze = bb_width < bb_width_20th  # low volatility regime
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h ADX(14)
    def calculate_adx(high, low, close, period=14):
        n = len(high)
        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
        
        # Wilder's smoothing
        atr = np.zeros(n)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)
        dx = np.zeros(n)
        
        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = 100 * (np.mean(plus_dm[i-period+1:i+1]) / atr[i])
                minus_di[i] = 100 * (np.mean(minus_dm[i-period+1:i+1]) / atr[i])
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(n)
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(sma[i]) or np.isnan(std[i]) or 
            np.isnan(bb_width_20th[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when 12h ADX > 25 (strong trend)
        strong_trend = adx_12h_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > upper[i-1] and squeeze[i-1]  # break above upper band after squeeze
        short_breakout = close[i] < lower[i-1] and squeeze[i-1]  # break below lower band after squeeze
        
        # Exit conditions: opposite breakout or loss of squeeze (volatility expansion)
        long_exit = short_breakout or not squeeze[i]
        short_exit = long_breakout or not squeeze[i]
        
        if long_breakout and strong_trend and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and strong_trend and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_bb_squeeze_breakout_adx_v1"
timeframe = "6h"
leverage = 1.0