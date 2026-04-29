#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(2) mean reversion with 4h trend filter and session filter
# Long when RSI(2) < 10 and price > 4h EMA50 (bullish regime)
# Short when RSI(2) > 90 and price < 4h EMA50 (bearish regime)
# Uses 4h EMA50 for trend filter to avoid mean reversion in strong trends
# Session filter (08-20 UTC) reduces noise trades
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
# Works in bull/bear: mean reversion in ranges, trend filter avoids false signals

name = "1h_RSI2_4hEMA50_MeanRev_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 55:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(2) on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[1:period+1])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smooth(gain, 2)
    avg_loss = wilders_smooth(loss, 2)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # warmup for RSI(2) and EMA50
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
            
        # Only trade during session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_rsi = rsi[i]
        curr_ema_50_4h = ema_50_4h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Mean reversion entries with trend filter
            if curr_rsi < 10 and curr_close > curr_ema_50_4h:
                # Bullish mean reversion in uptrend
                signals[i] = 0.20
                position = 1
            elif curr_rsi > 90 and curr_close < curr_ema_50_4h:
                # Bearish mean reversion in downtrend
                signals[i] = -0.20
                position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when RSI returns to neutral (50) or reverses
            if curr_rsi >= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position - exit conditions
            # Exit when RSI returns to neutral (50) or reverses
            if curr_rsi <= 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals