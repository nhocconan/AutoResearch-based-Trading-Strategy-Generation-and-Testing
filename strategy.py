#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA200 trend filter and volume spike confirmation
# Long when price breaks above 20-period Donchian high + price > 12h EMA200 (uptrend) + volume > 2x 20-period avg
# Short when price breaks below 20-period Donchian low + price < 12h EMA200 (downtrend) + volume > 2x 20-period avg
# Uses 12h EMA200 for primary trend direction to avoid counter-trend trades
# Volume spike filter reduces false breakouts during low-volatility periods
# Discrete position sizing (0.25) to control drawdown and minimize fee churn
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits
# Donchian breakouts work in both bull (continuation) and bear (mean reversion after panic) markets
# EMA200 filter ensures we only trade with the higher timeframe trend
# Volume confirmation ensures breakouts have conviction

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
    
    # Get 12h HTF data once before loop for EMA200 calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: EMA200 for trend direction ===
    close_12h = pd.Series(df_12h['close'].values)
    ema_200_12h = close_12h.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    # Calculate rolling max/min for Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need Donchian(20) + volume(20) + EMA200(200) buffer
    warmup = 220
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA (spike confirmation)
        vol_spike = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Trend filter: price relative to 12h EMA200
        uptrend = close[i] > ema_200_12h_aligned[i]
        downtrend = close[i] < ema_200_12h_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Uptrend (price > 12h EMA200)
        # 3. Volume spike confirmation
        if (close[i] > donchian_high[i]) and \
           uptrend and vol_spike:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Downtrend (price < 12h EMA200)
        # 3. Volume spike confirmation
        elif (close[i] < donchian_low[i]) and \
             downtrend and vol_spike:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_12hEMA200_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0