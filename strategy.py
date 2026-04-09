#!/usr/bin/env python3
# 4h_donchian_volume_chop_regime_v4
# Hypothesis: 4h Donchian breakout with volume confirmation and choppiness regime filter.
# Long: price breaks above Donchian(20) high + volume > 1.5x 20-period average + CHOP > 61.8 (range).
# Short: price breaks below Donchian(20) low + volume > 1.5x 20-period average + CHOP > 61.8.
# Uses daily HTF for trend filter: only long if price > daily EMA50, only short if price < daily EMA50.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 25-35 trades/year.
# Uses 1d HTF data for EMA50, called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volume_chop_regime_v4"
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
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_d = df_1d['close'].values
    ema_50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_d)
    
    # Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) - 14 period
    # CHOP = 100 * log10(sum(ATR(1)) / (log10(n) * max(high-n) - min(low-n)))
    # Simplified: CHOP = 100 * log10(ATR_sum / (log10(14) * (HHV - LLV))) / log10(14)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]  # First TR
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    highest_high = high_s.rolling(window=14, min_periods=14).max().values
    lowest_low = low_s.rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (highest_high - lowest_low))) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime: CHOP > 61.8 indicates ranging market (good for mean reversion/breakouts)
        chop_regime = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR loss of volume/chop confirmation
            if close[i] < donchian_low[i] or not (volume_confirmed and chop_regime):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR loss of volume/chop confirmation
            if close[i] > donchian_high[i] or not (volume_confirmed and chop_regime):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_regime:
                # Long entry: price breaks above Donchian high AND above daily EMA50 (uptrend filter)
                if close[i] > donchian_high[i] and close[i] > ema_50_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below daily EMA50 (downtrend filter)
                elif close[i] < donchian_low[i] and close[i] < ema_50_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals