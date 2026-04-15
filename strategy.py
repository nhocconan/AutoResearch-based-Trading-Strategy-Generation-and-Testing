#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian(20) upper + 1d EMA50 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian(20) lower + 1d EMA50 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.20) to minimize fee drag and control drawdown.
# 4h Donchian provides structure, 1d EMA50 filters whipsaws, volume confirms breakout strength.
# Session filter (08-20 UTC) reduces noise trades. Target: 15-35 trades/year per symbol.

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicator: Donchian Channel (20) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_period = 20
    upper_4h = pd.Series(high_4h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_4h = pd.Series(low_4h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # === 1d Indicator: EMA50 ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20, 20) + 5  # EMA50 + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper
        # 2. 1d EMA50 uptrend (close > EMA50)
        # 3. Volume confirmation
        if (close[i] > upper_4h_aligned[i]) and \
           (close[i] > ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower
        # 2. 1d EMA50 downtrend (close < EMA50)
        # 3. Volume confirmation
        elif (close[i] < lower_4h_aligned[i]) and \
             (close[i] < ema_50_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_4hDonchian20_1dEMA50_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0