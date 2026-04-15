#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above Donchian upper + 12h EMA50 uptrend + volume > 1.8x 20-period avg
# Short when price breaks below Donchian lower + 12h EMA50 downtrend + volume > 1.8x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 12h EMA50 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.8x) targets ~20-40 trades/year on 4h timeframe to avoid overtrading.
# Donchian channels calculated from prior 20 bars for structure-based entries.

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
    
    # === 4h Donchian Channel (20) ===
    # Upper = max(high over past 20 bars)
    # Lower = min(low over past 20 bars)
    # Using prior bar's data to avoid look-ahead
    roll_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    roll_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # EMA50 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. 12h EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (close[i] > roll_high[i]) and \
           (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. 12h EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (close[i] < roll_low[i]) and \
             (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0