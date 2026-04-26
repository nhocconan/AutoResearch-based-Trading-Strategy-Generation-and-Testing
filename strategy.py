#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_ChopFilter_v2
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and choppiness regime filter.
- Long when price breaks above Donchian(20) high AND 1d EMA50 uptrend AND chop < 61.8 (trending regime)
- Short when price breaks below Donchian(20) low AND 1d EMA50 downtrend AND chop < 61.8 (trending regime)
- Uses choppiness index to avoid false breakouts in ranging markets
- Exit on opposite Donchian level or trend reversal
- Designed for moderate frequency (target 20-50 trades/year on 4h) to minimize fee drag
- Novelty: Adding choppiness regime filter to Donchian breakout reduces whipsaws in bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate choppiness index on primary timeframe (4h)
    # Chop = 100 * log10(sum(ATR(14)) / log10(range(period))) / log10(period)
    # Simplified: Chop = 100 * log10(sum(True Range over period) / (max(high) - min(low))) / log10(period)
    period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    range_hl = max_high - min_low
    
    # Avoid division by zero
    chop = np.zeros(n)
    mask = (range_hl > 0) & (~np.isnan(atr_sum)) & (~np.isnan(range_hl))
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(period)
    # For invalid values, set to 50 (neutral)
    chop[~mask] = 50.0
    
    # Chop < 61.8 indicates trending regime (good for breakouts)
    trending_regime = chop < 61.8
    
    # Calculate Donchian channels on primary timeframe (4h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for Donchian, 14 for chop)
    start_idx = max(50, donchian_period, period)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_1d[i]) or np.isnan(trending_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with trend and regime filter
        if position == 0:
            # Long: Price breaks above Donchian high AND 1d uptrend AND trending regime
            if close[i] > donchian_high[i] and trend_1d[i] == 1 and trending_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1d downtrend AND trending regime
            elif close[i] < donchian_low[i] and trend_1d[i] == -1 and trending_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Donchian low OR 1d trend turns down
            if close[i] < donchian_low[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Donchian high OR 1d trend turns up
            if close[i] > donchian_high[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0