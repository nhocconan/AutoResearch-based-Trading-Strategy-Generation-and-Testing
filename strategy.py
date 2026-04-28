#!/usr/bin/env python3
# Hypothesis: 6h KAMA with 1-day RSI regime filter and volume confirmation.
# KAMA (Kaufman Adaptive Moving Average) adapts its smoothing based on market noise,
# making it responsive in trends and smooth in ranges. Combined with daily RSI
# (RSI > 60 for bullish regime, RSI < 40 for bearish regime) to filter trades
# in the direction of the higher timeframe momentum. Volume confirmation ensures
# breakouts have participation. Designed for 6h timeframe to target 50-150 total
# trades over 4 years (12-37/year). Works in both bull and bear markets by
# adapting to market conditions and filtering for clear momentum regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close)).cumsum()
    volatility = np.diff(volatility, prepend=volatility[0])
    
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RSI regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA on 6h data
    kama = _kama(close, er_length=10, fast_sc=2, slow_sc=30)
    
    # Volume filter: volume > 1.3x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: RSI > 60 = bullish, RSI < 40 = bearish
        bullish_regime = rsi_1d_aligned[i] > 60
        bearish_regime = rsi_1d_aligned[i] < 40
        
        # KAMA signals: price above KAMA = bullish, below = bearish
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Entry conditions with volume confirmation
        long_entry = bullish_regime and price_above_kama and volume_filter[i]
        short_entry = bearish_regime and price_below_kama and volume_filter[i]
        
        # Exit conditions: when regime changes or price crosses KAMA
        long_exit = (not bullish_regime) or (not price_above_kama)
        short_exit = (not bearish_regime) or (not price_below_kama)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_KAMA_1dRSI_RegimeFilter_Volume"
timeframe = "6h"
leverage = 1.0