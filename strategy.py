#!/usr/bin/env python3
"""
6h_Poisson_Regime_Momentum
Hypothesis: In 6h timeframe, use Poisson-based volatility regime detection (low vol = mean reversion, high vol = momentum) combined with 1-week trend filter. 
Poisson process models rare volatility spikes; low lambda indicates stability (favor mean reversion), high lambda indicates instability (favor momentum).
Trades only when 1-week trend aligns with signal direction. Designed for low-frequency, high-conviction trades in both bull and bear markets.
Target: 20-50 trades/year per symbol to avoid fee drag.
"""

name = "6h_Poisson_Regime_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import exp, factorial
from mtf_data import get_htf_data, align_htf_to_ltf

def poisson_probability(k, lam):
    """Compute P(X=k) for Poisson distribution"""
    if lam <= 0:
        return 0.0
    return (lam**k * exp(-lam)) / factorial(k)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 6-period returns for volatility proxy
    returns = np.diff(np.log(close), prepend=0)
    
    # Poisson lambda estimation: frequency of large returns (>2% absolute)
    large_move = np.abs(returns) > 0.02
    lambda_est = np.full(n, np.nan)
    window = 28  # ~1 week of 6h bars
    
    for i in range(window, n):
        # Count large moves in window
        k = np.sum(large_move[i-window:i])
        lambda_est[i] = k  # MLE for Poisson lambda is sample mean
    
    # Regime classification: low lambda = mean reversion regime, high lambda = momentum regime
    # Threshold: lambda < 2 = low vol (mean reversion), lambda >= 2 = high vol (momentum)
    regime_mean_revert = lambda_est < 2.0
    regime_momentum = lambda_est >= 2.0
    
    # 6h RSI for mean reversion signals
    delta = np.diff(close, prepend=0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder smoothing for RSI
    def wilders_smoothing(x, period):
        smoothed = np.full_like(x, np.nan)
        if len(x) < period:
            return smoothed
        smoothed[period-1] = np.mean(x[:period])
        for i in range(period, len(x)):
            smoothed[i] = (smoothed[i-1] * (period-1) + x[i]) / period
        return smoothed
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h momentum: price change over 3 periods
    mom = np.zeros_like(close)
    mom[3:] = (close[3:] - close[:-3]) / close[:-3]
    
    # 1-week trend filter: EMA crossover
    close_1w = df_1w['close'].values
    ema_fast = np.zeros_like(close_1w)
    ema_slow = np.zeros_like(close_1w)
    
    # Fast EMA 8
    alpha_fast = 2 / (8 + 1)
    ema_fast[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_fast[i] = alpha_fast * close_1w[i] + (1 - alpha_fast) * ema_fast[i-1]
    
    # Slow EMA 21
    alpha_slow = 2 / (21 + 1)
    ema_slow[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        ema_slow[i] = alpha_slow * close_1w[i] + (1 - alpha_slow) * ema_slow[i-1]
    
    # Trend direction: 1 = uptrend, -1 = downtrend, 0 = unclear
    trend_1w = np.zeros_like(close_1w)
    trend_1w[ema_fast > ema_slow] = 1
    trend_1w[ema_fast < ema_slow] = -1
    
    # Align 1w trend to 6h
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 12  # Prevent overtrading (approx 3 days)
    
    start_idx = max(28, 14, 3)  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(lambda_est[i]) or np.isnan(rsi[i]) or 
            np.isnan(mom[i]) or np.isnan(trend_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Mean reversion regime: RSI extremes
            if regime_mean_revert[i]:
                if rsi[i] < 30 and trend_1w_aligned[i] == 1:  # Oversold in uptrend
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                elif rsi[i] > 70 and trend_1w_aligned[i] == -1:  # Overbought in downtrend
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
            # Momentum regime: price momentum continuation
            elif regime_momentum[i]:
                if mom[i] > 0.015 and trend_1w_aligned[i] == 1:  # Strong up momentum in uptrend
                    signals[i] = 0.25
                    position = 1
                    bars_since_last_trade = 0
                elif mom[i] < -0.015 and trend_1w_aligned[i] == -1:  # Strong down momentum in downtrend
                    signals[i] = -0.25
                    position = -1
                    bars_since_last_trade = 0
        elif position == 1:
            # Exit long: RSI overbought or momentum fails
            exit_signal = (rsi[i] > 70) or (mom[i] < -0.005)
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI oversold or momentum fails
            exit_signal = (rsi[i] < 30) or (mom[i] > 0.005)
            if exit_signal:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals