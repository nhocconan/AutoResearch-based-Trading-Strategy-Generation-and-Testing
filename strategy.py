#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA(8,21) crossover with 4h Supertrend(10,3) trend filter and volume confirmation
# Long when 1h EMA8 crosses above EMA21 + price > 4h Supertrend + volume > 1.5x 20-period avg
# Short when 1h EMA8 crosses below EMA21 + price < 4h Supertrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-35 trades/year.
# Supertrend provides objective trend direction, avoiding whipsaws in sideways markets.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring price relative to Supertrend.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicator: Supertrend(10,3) ===
    atr_period = 10
    multiplier = 3.0
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = high_4h[0] - low_4h[0]
    tr2[0] = np.abs(high_4h[0] - close_4h[0])
    tr3[0] = np.abs(low_4h[0] - close_4h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR using Wilder's smoothing
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate Supertrend
    hl2 = (high_4h + low_4h) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upperband[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = upperband[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = lowerband[i]
        elif direction[i] == 1 and upperband[i] < supertrend[i-1]:
            supertrend[i] = upperband[i]
        elif direction[i] == -1 and lowerband[i] > supertrend[i-1]:
            supertrend[i] = lowerband[i]
        elif direction[i] == 1:
            supertrend[i] = max(upperband[i], supertrend[i-1])
        else:
            supertrend[i] = min(lowerband[i], supertrend[i-1])
    
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # === 1h Indicators: EMA(8) and EMA(21) ===
    ema_fast = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === 1h Volume SMA(20) for confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(21, 20, atr_period*2) + 5  # EMA(21) + Vol(20) + Supertrend buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Fast EMA crosses above slow EMA (bullish crossover)
        # 2. Price above 4h Supertrend (uptrend confirmation)
        # 3. Volume confirmation
        if (ema_fast[i] > ema_slow[i]) and (ema_fast[i-1] <= ema_slow[i-1]) and \
           (close[i] > supertrend_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Fast EMA crosses below slow EMA (bearish crossover)
        # 2. Price below 4h Supertrend (downtrend confirmation)
        # 3. Volume confirmation
        elif (ema_fast[i] < ema_slow[i]) and (ema_fast[i-1] >= ema_slow[i-1]) and \
             (close[i] < supertrend_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA8_21_4hSupertrend10_3_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0