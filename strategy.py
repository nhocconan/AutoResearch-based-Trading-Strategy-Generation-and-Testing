#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with 1d Williams %R mean reversion filter and volume confirmation
# Long when KAMA(10,2,30) slope > 0 (uptrend) + 1d Williams %R(14) < -80 (oversold pullback) + volume > 1.5x 20-period avg
# Short when KAMA slope < 0 (downtrend) + 1d Williams %R(14) > -20 (overbought rally) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# KAMA adapts to market noise, reducing false signals in choppy conditions. 1d Williams %R ensures we enter on higher timeframe extremes.
# Works in bull markets (buying 1d oversold in uptrend) and bear markets (selling 1d overbought in downtrend).

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
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # === 1d Indicator: Williams %R (14-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    lookback_1d = 14
    highest_high_1d = pd.Series(high_1d).rolling(window=lookback_1d, min_periods=lookback_1d).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=lookback_1d, min_periods=lookback_1d).min().values
    denominator_1d = highest_high_1d - lowest_low_1d
    williams_r_1d = np.where(denominator_1d != 0, -100 * (highest_high_1d - close_1d) / denominator_1d, -50)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # === 4h Indicator: KAMA (10,2,30) ===
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility correctly
    volatility = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fastest = 2 / (2 + 1)   # EMA(2)
    slowest = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fastest - slowest) + slowest) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    # Slope of KAMA (trend direction)
    kama_slope = np.diff(kama, prepend=kama[0])
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 10  # KAMA(30) + volume(20) + ER(10)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high_1d[i]) or np.isnan(lowest_low_1d[i]) or
            np.isnan(williams_r_1d_aligned[i]) or np.isnan(vol_sma_20[i]) or
            np.isnan(kama_slope[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. KAMA slope > 0 (uptrend)
        # 2. 1d Williams %R < -80 (oversold pullback)
        # 3. Volume confirmation
        if (kama_slope[i] > 0) and \
           (williams_r_1d_aligned[i] < -80) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. KAMA slope < 0 (downtrend)
        # 2. 1d Williams %R > -20 (overbought rally)
        # 3. Volume confirmation
        elif (kama_slope[i] < 0) and \
             (williams_r_1d_aligned[i] > -20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_KAMA10_2_30_1dWilliamsR14_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0