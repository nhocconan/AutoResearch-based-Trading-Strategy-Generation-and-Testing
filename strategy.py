#!/usr/bin/env python3
"""
6h_LiquidityPool_Reversion
Hypothesis: In BTC/ETH, price tends to revert from liquidity pools (equal highs/lows) on 6h timeframe.
- Identifies equal highs/lows (within 0.2%) as liquidity pools
- Enters mean reversion when price touches these pools with RSI < 30 (long) or > 70 (short)
- Uses 12h trend filter (EMA50) to avoid counter-trend traps
- Works in bull/bear via trend alignment and mean reversion at institutional levels
- Target: 50-150 trades over 4 years (12-37/year) with strict entry conditions
"""

name = "6h_LiquidityPool_Reversion"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # RSI(14) for mean reversion signals
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices, prepend=prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Wilder's smoothing
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[1:period+1])
        avg_loss[period] = np.mean(loss[1:period+1])
        
        for i in range(period+1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Identify liquidity pools (equal highs/lows within 0.2%)
    def find_liquidity_pools(high, low, lookback=20, threshold=0.002):
        """Find equal highs/lows within threshold percentage"""
        n = len(high)
        liquidity_high = np.full(n, np.nan)
        liquidity_low = np.full(n, np.nan)
        
        for i in range(lookback, n):
            # Check for equal highs in lookback window
            window_high = high[i-lookback:i]
            max_high = np.max(window_high)
            # Find if current high or recent high is within threshold of max
            if high[i] >= max_high * (1 - threshold) or max_high >= high[i] * (1 - threshold):
                liquidity_high[i] = max_high
            
            # Check for equal lows in lookback window
            window_low = low[i-lookback:i]
            min_low = np.min(window_low)
            # Find if current low or recent low is within threshold of min
            if low[i] <= min_low * (1 + threshold) or min_low <= low[i] * (1 + threshold):
                liquidity_low[i] = min_low
        
        return liquidity_high, liquidity_low
    
    liq_high, liq_low = find_liquidity_pools(high, low, lookback=20, threshold=0.002)
    
    # Align liquidity pools to current timeframe (they're already on 6h)
    # No alignment needed as we calculated on the same timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above EMA50 = uptrend, below = downtrend
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price at liquidity low (support) + oversold RSI + uptrend
            if (not np.isnan(liq_low[i]) and 
                abs(low[i] - liq_low[i]) / liq_low[i] < 0.005 and  # Within 0.5% of liquidity low
                rsi[i] < 30 and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price at liquidity high (resistance) + overbought RSI + downtrend
            elif (not np.isnan(liq_high[i]) and 
                  abs(high[i] - liq_high[i]) / liq_high[i] < 0.005 and  # Within 0.5% of liquidity high
                  rsi[i] > 70 and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit when RSI reaches neutral (50) or breaks below liquidity low
            if rsi[i] >= 50 or (not np.isnan(liq_low[i]) and low[i] < liq_low[i] * 0.995):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit when RSI reaches neutral (50) or breaks above liquidity high
            if rsi[i] <= 50 or (not np.isnan(liq_high[i]) and high[i] > liq_high[i] * 1.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals