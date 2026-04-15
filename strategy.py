#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA50 (uptrend regime)
# Short when Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND price < 1d EMA50 (downtrend regime)
# Uses Elder Ray (Bull/Bear Power) to measure bull/bear strength relative to EMA13
# 1d EMA50 provides regime filter to avoid counter-trend trades
# Discrete sizing 0.25 to limit drawdown and minimize fee churn
# Target: 50-150 total trades over 4 years on BTC/ETH/SOL

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 for regime filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Elder Ray Components ===
    # EMA13 as the reference
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13) + 5  # EMA50 + EMA13 + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (strong bullish momentum)
        # 2. Bear Power < 0 (weak bearish momentum)
        # 3. Price > 1d EMA50 (uptrend regime)
        if (bull_power[i] > 0) and (bear_power[i] < 0) and (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (strong bearish momentum)
        # 2. Bull Power > 0 (weak bullish momentum)
        # 3. Price < 1d EMA50 (downtrend regime)
        elif (bear_power[i] < 0) and (bull_power[i] > 0) and (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_1dEMA50_Regime_Filter_v1"
timeframe = "6h"
leverage = 1.0