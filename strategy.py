#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with volume confirmation and 1d EMA trend filter
# Long when price breaks above 12h Donchian upper (20) + volume > 2.0x 20-period avg + 1d EMA50 > EMA200 (uptrend)
# Short when price breaks below 12h Donchian lower (20) + volume > 2.0x 20-period avg + 1d EMA50 < EMA200 (downtrend)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-25/year).
# Donchian channels provide objective breakout levels. EMA filter ensures we trade with the higher timeframe trend.
# Works in bull markets (trend continuation) and bear markets (strong downtrends) by requiring EMA alignment.

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
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # === 1d Indicator: EMA50 and EMA200 (trend filter) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    # For Donchian, we need to calculate on 12h data directly
    # Since we're on 12h timeframe, we can use the prices directly
    high_12h = high
    low_12h = low
    
    # Calculate Donchian channels: upper = max(high, lookback), lower = min(low, lookback)
    lookback = 20
    donchian_upper = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(100, 200)  # EMA200 needs 200 periods
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema50_aligned[i]) or np.isnan(ema200_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
        is_uptrend = ema50_aligned[i] > ema200_aligned[i]
        is_downtrend = ema50_aligned[i] < ema200_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian upper (20)
        # 2. Uptrend on 1d (EMA50 > EMA200)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and \
           is_uptrend and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian lower (20)
        # 2. Downtrend on 1d (EMA50 < EMA200)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and \
             is_downtrend and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_Volume_EMA50_200_Trend_Filter_v1"
timeframe = "12h"
leverage = 1.0