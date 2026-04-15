#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1w trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + 1w close > 1w EMA50 (uptrend) + volume > 2x 20-period avg
# Short when price breaks below Donchian(20) low + 1w close < 1w EMA50 (downtrend) + volume > 2x 20-period avg
# Uses discrete position sizing (0.30) to balance risk and reward. Designed for low trade frequency (20-40/year).
# Donchian channels provide clear breakout levels. 1w EMA50 filter ensures we trade with the higher timeframe trend.
# Volume confirmation avoids false breakouts. Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend).

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w Indicator: EMA50 (trend filter) ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(lookback, 50) + 20  # Donchian(20) + EMA50(50) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper channel
        # 2. 1w close > 1w EMA50 (uptrend)
        # 3. Volume confirmation
        if (close[i] > highest_high[i]) and \
           (close_1w_aligned := align_htf_to_ltf(prices, df_1w, close_1w)[i]) > ema50_1w_aligned[i] and vol_confirm:
            signals[i] = 0.30
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower channel
        # 2. 1w close < 1w EMA50 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < lowest_low[i]) and \
             (close_1w_aligned := align_htf_to_ltf(prices, df_1w, close_1w)[i]) < ema50_1w_aligned[i] and vol_confirm:
            signals[i] = -0.30
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1wEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0