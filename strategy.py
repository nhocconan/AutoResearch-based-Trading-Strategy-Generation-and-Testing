#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with volume spike and 12h EMA trend filter
# Long when Williams %R(14) crosses above -80 (oversold reversal) + volume > 2.0x 20-period avg + price > 12h EMA50
# Short when Williams %R(14) crosses below -20 (overbought reversal) + volume > 2.0x 20-period avg + price < 12h EMA50
# Uses 4h price momentum (Williams %R) and 12h EMA for multi-timeframe trend alignment
# Designed for low trade frequency (15-25/year) to minimize fee drag while capturing reversals in both bull and bear markets
# Volume spike filter ensures participation, reducing false signals
# Works in ranging markets via mean reversion and in trending markets via trend filter alignment

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 4h Indicators: Williams %R(14) ===
    highest_high_4h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r_4h = -100 * (highest_high_4h - close) / (highest_high_4h - lowest_low_4h)
    # Handle division by zero (when high == low)
    williams_r_4h = np.where((highest_high_4h - lowest_low_4h) == 0, -50, williams_r_4h)
    
    # === 12h Indicator: EMA50 ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 4h timeframe
    williams_r_4h_aligned = align_htf_to_ltf(prices, df_4h, williams_r_4h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_4h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Previous Williams %R for crossover detection
        prev_williams_r = williams_r_4h_aligned[i-1]
        curr_williams_r = williams_r_4h_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Williams %R crosses above -80 (oversold reversal)
        # 2. Volume confirmation
        # 3. Price above 12h EMA50 (uptrend filter)
        if (prev_williams_r <= -80 and curr_williams_r > -80) and vol_confirm and \
           (close[i] > ema_50_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R crosses below -20 (overbought reversal)
        # 2. Volume confirmation
        # 3. Price below 12h EMA50 (downtrend filter)
        elif (prev_williams_r >= -20 and curr_williams_r < -20) and vol_confirm and \
             (close[i] < ema_50_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR_VolumeSpike_12hEMA50_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0