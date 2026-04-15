#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h EMA50 trend filter with volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power rising (less negative) + 12h EMA50 uptrend + volume > 1.5x 20-period avg
# Short when Bear Power < 0 AND Bull Power falling (less positive) + 12h EMA50 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 12h EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~50-150 trades over 4 years to minimize fee drag on 6h timeframe.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: EMA50 ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Higher highs relative to trend
    bear_power = low - ema_13   # Lower lows relative to trend
    
    # Rate of change of Elder Ray powers (to detect momentum shift)
    bull_power_roc = np.zeros_like(bull_power)
    bear_power_roc = np.zeros_like(bear_power)
    bull_power_roc[1:] = (bull_power[1:] - bull_power[:-1]) / np.abs(bull_power[:-1] + 1e-10)
    bear_power_roc[1:] = (bear_power[1:] - bear_power[:-1]) / np.abs(bear_power[:-1] + 1e-10)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 13, 20) + 5  # EMA50(12h) + EMA13 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_roc[i]) or 
            np.isnan(bear_power_roc[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Bull Power > 0 (price above EMA13 on highs)
        # 2. Bear Power rising (becoming less negative) - indicating weakening bearish momentum
        # 3. 12h EMA50 uptrend (close > EMA50)
        # 4. Volume confirmation
        if (bull_power[i] > 0) and \
           (bear_power_roc[i] > 0) and \
           (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Bear Power < 0 (price below EMA13 on lows)
        # 2. Bull Power falling (becoming less positive) - indicating weakening bullish momentum
        # 3. 12h EMA50 downtrend (close < EMA50)
        # 4. Volume confirmation
        elif (bear_power[i] < 0) and \
             (bull_power_roc[i] < 0) and \
             (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_12hEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0