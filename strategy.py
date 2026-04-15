#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend following with 1d RSI filter and volume confirmation
# Long when KAMA(ER=10) is rising + RSI(14) from 1d < 40 (oversold bounce) + volume > 1.5x 20-period avg
# Short when KAMA(ER=10) is falling + RSI(14) from 1d > 60 (overbought pullback) + volume > 1.5x 20-period avg
# KAMA adapts to market noise, reducing whipsaw in choppy markets. RSI filter avoids buying strength/selling weakness.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend) via adaptive trend.
# Designed for low trade frequency (20-40/year) to minimize fee drag.

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
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[rsi_period] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period + 1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])  # align length with df_1d
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 4h Indicator: KAMA (ER=10, fast=2, slow=30) ===
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if hasattr(np.sum, 'axis') else np.sum(np.abs(np.diff(close)))
    # Correct volatility calculation: sum of absolute changes over er_period window
    volatility = np.zeros_like(close)
    for i in range(er_period, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i])))
    
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (fast_sc/slow_sc - 1) + 1) ** 2
    
    kama = np.zeros_like(close)
    kama[er_period] = close[er_period]  # seed
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 if rising, -1 if falling, 0 if flat (using 3-period slope)
    kama_dir = np.zeros_like(kama)
    for i in range(3, len(kama)):
        if kama[i] > kama[i-3]:
            kama_dir[i] = 1
        elif kama[i] < kama[i-3]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = 0
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(er_period + 3, 20) + 5  # KAMA seed + direction + volume + RSI
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. KAMA is rising (trend up)
        # 2. 1d RSI < 40 (oversold bounce opportunity)
        # 3. Volume confirmation
        if (kama_dir[i] == 1) and \
           (rsi_aligned[i] < 40) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. KAMA is falling (trend down)
        # 2. 1d RSI > 60 (overbought pullback opportunity)
        # 3. Volume confirmation
        elif (kama_dir[i] == -1) and \
             (rsi_aligned[i] > 60) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_KAMA10_1dRSI40_60_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0