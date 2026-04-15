#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend + RSI mean reversion + volume spike filter
# Long when KAMA up + RSI < 30 (oversold) + volume > 1.5x 20-period avg
# Short when KAMA down + RSI > 70 (overbought) + volume > 1.5x 20-period avg
# Uses discrete sizing (0.25) to control drawdown and fee drag.
# KAMA adapts to market noise, reducing whipsaws in ranging markets.
# RSI extremes provide mean reversion entries within the trend.
# Volume spike confirms participation, filtering low-quality signals.
# Target: 15-25 trades/year on 12h timeframe to minimize fee impact.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: RSI(14) ===
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # === 12h KAMA(10,2,30) ===
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, n=1, prepend=close[0:1])), axis=0)
    # Fix volatility calculation for streaming
    volatility_series = pd.Series(close).diff().abs().rolling(window=10, min_periods=1).sum().values
    er = np.where(volatility_series > 0, change / volatility_series, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 12h Volume SMA(20) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20) + 5  # RSI(14) + KAMA seed + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(vol_sma_20[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price above KAMA (uptrend)
        # 2. RSI < 30 (oversold)
        # 3. Volume confirmation
        if (close[i] > kama[i]) and \
           (rsi_14_aligned[i] < 30) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price below KAMA (downtrend)
        # 2. RSI > 70 (overbought)
        # 3. Volume confirmation
        elif (close[i] < kama[i]) and \
             (rsi_14_aligned[i] > 70) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_KAMA10_2_30_1dRSI14_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0