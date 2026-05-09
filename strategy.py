#!/usr/bin/env python3
# Hypothesis: 1-day KAMA trend with 1-week EMA trend filter and volume confirmation
# Long when KAMA is rising and price > KAMA, with 1-week EMA50 > EMA200 (bullish regime) and volume > 1.5x average
# Short when KAMA is falling and price < KAMA, with 1-week EMA50 < EMA200 (bearish regime) and volume > 1.5x average
# Exit when price crosses back through KAMA
# Uses adaptive trend (KAMA) with higher timeframe regime filter to avoid false signals
# Target: 30-100 total trades over 4 years (7-25/year) with size 0.25

name = "1d_KAMA_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive moving average)
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1) if len(close) > 1 else np.array([])
    # Pad volatility to match change length
    if len(volatility) > 0:
        volatility = np.concatenate([np.full(er_period-1, np.nan), volatility])
    else:
        volatility = np.full_like(change, np.nan)
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    er = np.concatenate([np.full(er_period-1, np.nan), er])
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    sc = np.where(np.isnan(sc), 0, sc)
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    if len(close) > er_period:
        kama[er_period] = close[er_period]  # Seed
        for i in range(er_period + 1, len(close)):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # Get 1-week EMA data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA50 and EMA200 on weekly data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False).mean().values
    
    # Regime: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
    regime_bullish = ema50_1w > ema200_1w
    regime_bearish = ema50_1w < ema200_1w
    
    # Align regime to daily timeframe
    regime_bullish_aligned = align_htf_to_ltf(prices, df_1w, regime_bullish.astype(float))
    regime_bearish_aligned = align_htf_to_ltf(prices, df_1w, regime_bearish.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(regime_bullish_aligned[i]) or 
            np.isnan(regime_bearish_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: KAMA rising, price > KAMA, bullish regime, volume confirmation
            if (close[i] > kama[i] and 
                i > 0 and close[i-1] <= kama[i-1] and  # KAMA just turned up / price crossed above
                regime_bullish_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, price < KAMA, bearish regime, volume confirmation
            elif (close[i] < kama[i] and 
                  i > 0 and close[i-1] >= kama[i-1] and  # KAMA just turned down / price crossed below
                  regime_bearish_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals