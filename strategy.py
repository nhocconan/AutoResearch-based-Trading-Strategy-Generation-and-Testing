#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper channel (20-period high) + 1w close > 1w EMA50 (uptrend) + volume > 1.5x 20-period avg
# Short when price breaks below 12h Donchian lower channel (20-period low) + 1w close < 1w EMA50 (downtrend) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Donchian channels provide clear breakout levels. 1w EMA50 filter ensures we trade with the higher timeframe trend.
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend).

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
    
    # === 12h Donchian Channel (20-period) ===
    period_donch = 20
    highest_high = pd.Series(high).rolling(window=period_donch, min_periods=period_donch).max().values
    lowest_low = pd.Series(low).rolling(window=period_donch, min_periods=period_donch).min().values
    upper_channel = highest_high
    lower_channel = lowest_low
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need max of: donchian(20) + volume(20)
    warmup = 20 + 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1w close for trend filter
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper channel (close > upper_channel)
        # 2. 1w close > 1w EMA50 (uptrend)
        # 3. Volume confirmation
        if (close[i] > upper_channel[i]) and \
           (close_1w_aligned > ema50_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower channel (close < lower_channel)
        # 2. 1w close < 1w EMA50 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < lower_channel[i]) and \
             (close_1w_aligned < ema50_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1wEMA50_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0