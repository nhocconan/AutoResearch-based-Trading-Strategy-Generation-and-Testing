#!/usr/bin/env python3
# 4h_donchian_breakout_volume_chop_v3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.5x 20-period average) and chop regime filter (CHOP(14) between 38.2 and 61.8 for ranging, >61.8 for mean reversion). Long when price breaks above Donchian upper band in choppy market (>61.8), short when breaks below lower band. Uses discrete position sizing (0.25) to minimize fee churn. ATR trailing stop (2.0x) protects against reversals. Designed for both bull and bear markets via regime adaptation.
# Timeframe: 4h, HTF: none needed (uses 14-period CHOP on same timeframe)
# Target: 20-50 trades/year (80-200 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd

name = "4h_donchian_breakout_volume_chop_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(atr14) / (n * (max(high)n - min(low)n))) / log10(n)
    # Where n=14, sum(atr14) is rolling sum of true range over 14 periods
    # max(high)n is rolling max of high over 14 periods
    # min(low)n is rolling min of low over 14 periods
    atr_for_chop = tr  # true range already calculated
    sum_atr14 = pd.Series(atr_for_chop).rolling(window=14, min_periods=14).sum().values
    max_high14 = high_s.rolling(window=14, min_periods=14).max().values
    min_low14 = low_s.rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = 14 * (max_high14 - min_low14)
    chop_raw = np.where(denominator != 0, sum_atr14 / denominator, 1.0)
    chop = 100 * np.log10(np.maximum(chop_raw, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Price breaks below Donchian lower band
            elif low[i] < donchian_low[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Price breaks above Donchian upper band
            elif high[i] > donchian_high[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above Donchian upper band, volume confirmed, and chop > 61.8 (trending or choppy enough for breakout)
            if (high[i] > donchian_high[i] and volume_confirmed and chop[i] > 61.8):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower band, volume confirmed, and chop > 61.8
            elif (low[i] < donchian_low[i] and volume_confirmed and chop[i] > 61.8):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals