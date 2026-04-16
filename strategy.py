#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and weekly trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + weekly close > weekly EMA50
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + weekly close < weekly EMA50
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Weekly EMA50 filter ensures we only trade with the higher timeframe trend
# Volume confirmation reduces false breakouts

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
    
    # Get 1d HTF data once before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Donchian Channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min()
    
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # Align Donchian levels to 12h timeframe (wait for 1d bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1w Indicator: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d data for Donchian (20) + volume(20) + buffer
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Weekly trend filter: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema_50_aligned[i]
        weekly_downtrend = close[i] < ema_50_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period)
        # 2. Volume confirmation
        # 3. Weekly uptrend (price > weekly EMA50)
        if (close[i] > donchian_high_aligned[i]) and \
           vol_confirm and weekly_uptrend:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period)
        # 2. Volume confirmation
        # 3. Weekly downtrend (price < weekly EMA50)
        elif (close[i] < donchian_low_aligned[i]) and \
             vol_confirm and weekly_downtrend:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1wEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0