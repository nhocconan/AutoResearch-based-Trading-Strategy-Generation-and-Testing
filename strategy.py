#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend direction + RSI(2) mean reversion + volume spike filter
# Long when: KAMA rising (bullish trend) + RSI(2) < 10 (extreme oversold) + volume > 2.0x 20-day avg
# Short when: KAMA falling (bearish trend) + RSI(2) > 90 (extreme overbought) + volume > 2.0x 20-day avg
# Uses discrete position sizing (0.30) to minimize fee churn. Designed for low trade frequency (10-25/year).
# KAMA adapts to market noise, reducing whipsaws in choppy markets. RSI(2) captures short-term mean reversion.
# Volume spike confirms institutional participation. Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicator: KAMA (trend direction) ===
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)[:len(change)]
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === 1d Indicator: RSI(2) ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[1] = np.mean(gain[:1]) if len(gain) >= 1 else 0
    avg_loss[1] = np.mean(loss[:1]) if len(loss) >= 1 else 0
    
    # Wilder's smoothing
    for i in range(2, n):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i-1]) / 2
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i-1]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1d Indicator: Volume SMA(20) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 2, 20) + 5  # KAMA(30) + RSI(2) + volume(20)
    
    for i in range(warmup, n):
        # Volume filter: current volume > 2.0x 20-day volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. KAMA rising (bullish trend)
        # 2. RSI(2) < 10 (extreme oversold)
        # 3. Volume confirmation
        if (kama[i] > kama[i-1]) and \
           (rsi[i] < 10) and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. KAMA falling (bearish trend)
        # 2. RSI(2) > 90 (extreme overbought)
        # 3. Volume confirmation
        elif (kama[i] < kama[i-1]) and \
             (rsi[i] > 90) and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_KAMA_RSI2_Volume_Spike_v1"
timeframe = "1d"
leverage = 1.0