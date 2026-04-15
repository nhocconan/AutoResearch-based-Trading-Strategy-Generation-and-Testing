#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover with 4h trend filter and volume confirmation
# Long when 1h EMA12 crosses above EMA26, 4h EMA50 is rising (trend up), and volume > 1.5x 20-period avg
# Short when 1h EMA12 crosses below EMA26, 4h EMA50 is falling (trend down), and volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee churn. Session filter (08-20 UTC) reduces noise.
# Designed for low trade frequency (15-35/year) by requiring EMA crossover + 4h trend alignment + volume spike.
# Works in bull markets (trend following) and bear markets (trend continuation) by aligning with 4h EMA50 direction.

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
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: EMA50 (trend direction) ===
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h EMA50 slope (rising/falling) - using 3-bar change for trend confirmation
    ema_50_slope_4h = np.diff(ema_50_4h_aligned, prefill=0)
    ema_50_rising = ema_50_slope_4h > 0
    ema_50_falling = ema_50_slope_4h < 0
    
    # === 1h Indicators: EMA12, EMA26 ===
    ema_12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # EMA crossover signals
    ema_12_prev = np.roll(ema_12, 1)
    ema_26_prev = np.roll(ema_26, 1)
    ema_12_prev[0] = ema_12[0]
    ema_26_prev[0] = ema_26[0]
    
    ema_12_cross_above = (ema_12 > ema_26) & (ema_12_prev <= ema_26_prev)
    ema_12_cross_below = (ema_12 < ema_26) & (ema_12_prev >= ema_26_prev)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_12[i]) or np.isnan(ema_26[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. 1h EMA12 crosses above EMA26
        # 2. 4h EMA50 is rising (trend up)
        # 3. Volume confirmation
        if ema_12_cross_above[i] and ema_50_rising[i] and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. 1h EMA12 crosses below EMA26
        # 2. 4h EMA50 is falling (trend down)
        # 3. Volume confirmation
        elif ema_12_cross_below[i] and ema_50_falling[i] and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_EMA12_26_Crossover_4hEMA50_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0