#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation.
# In trending regimes (CHOP < 38.2): trade breakouts in trend direction using 4h EMA50.
# In ranging regimes (CHOP > 61.8): fade extremes at Donchian bands with mean reversion.
# Designed for low trade frequency (target: 20-40/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear markets via mean reversion in ranges.
name = "4h_ChopRegime_Donchian20_Volume"
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
    
    # Donchian(20) channels
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for Choppiness Index
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # True Range sum and ATR for denominator
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index: CHOP = 100 * log10(TR_sum / (ATR * 14)) / log10(14)
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    
    # Regime filters
    trending = chop < 38.2   # Trending regime
    ranging = chop > 61.8    # Ranging regime
    
    # EMA50 for trend direction
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up = close > ema50
    trend_down = close < ema50
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(chop[i]) or np.isnan(ema50[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            if trending[i]:
                # Trend-following: breakout in trend direction
                long_condition = (close[i] > highest_high_20[i]) and trend_up[i] and volume_spike[i]
                short_condition = (close[i] < lowest_low_20[i]) and trend_down[i] and volume_spike[i]
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
            elif ranging[i]:
                # Mean reversion: fade at Donchian extremes
                long_condition = (close[i] <= lowest_low_20[i]) and volume_spike[i]
                short_condition = (close[i] >= highest_high_20[i]) and volume_spike[i]
                
                if long_condition:
                    signals[i] = 0.20
                    position = 1
                elif short_condition:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            if trending[i]:
                # Exit trend long: close below EMA50 or Donchian low
                if close[i] < ema50[i] or close[i] < lowest_low_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Exit range long: return to mean or breakout
                if (close[i] >= (highest_high_20[i] + lowest_low_20[i]) / 2) or \
                   (close[i] > highest_high_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
        elif position == -1:
            if trending[i]:
                # Exit trend short: close above EMA50 or Donchian high
                if close[i] > ema50[i] or close[i] > highest_high_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Exit range short: return to mean or breakout
                if (close[i] <= (highest_high_20[i] + lowest_low_20[i]) / 2) or \
                   (close[i] < lowest_low_20[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals