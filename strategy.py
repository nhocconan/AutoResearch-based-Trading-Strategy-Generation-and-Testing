#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeRegime_v1
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and choppiness regime filter.
Only take long breakouts when price > 1d EMA50 and chop < 61.8 (trending regime).
Only take short breakouts when price < 1d EMA50 and chop < 61.8.
Volume confirmation requires volume > 1.5x 20-period average.
Designed for 20-50 trades/year (80-200 over 4 years) by requiring confluence of trend, regime, and volume.
Works in bull/bear via 1d trend filter: only takes longs in uptrend, shorts in downtrend.
Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend and chop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n+1))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                       np.maximum(np.abs(df_1d['high'].values - df_1d['close'].shift(1)),
                                  np.abs(df_1d['low'].values - df_1d['close'].shift(1))))
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(sum_atr_14 / (14 * np.log10(15))) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    chop_filter = chop_1d_aligned < 61.8  # True when trending (chop < 61.8)
    
    # Calculate 20-period volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) from 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 14 for chop, 20 for Donchian/volume)
    start_idx = max(50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend and regime filters
        trend_up = htf_trend[i] == 1
        trend_down = htf_trend[i] == -1
        regime_trending = chop_filter[i]
        
        # Breakout conditions
        if trend_up and regime_trending and volume_confirm:
            # Long breakout above Donchian high in uptrend + trending regime
            if close[i] > donchian_high[i]:
                if position != 1:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.25
            # Exit long if price falls below Donchian low (reversal signal)
            elif position == 1 and close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        elif trend_down and regime_trending and volume_confirm:
            # Short breakdown below Donchian low in downtrend + trending regime
            if close[i] < donchian_low[i]:
                if position != -1:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = -0.25
            # Exit short if price rises above Donchian high (reversal signal)
            elif position == -1 and close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
        else:
            # No clear signal: hold current position or stay flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeRegime_v1"
timeframe = "4h"
leverage = 1.0