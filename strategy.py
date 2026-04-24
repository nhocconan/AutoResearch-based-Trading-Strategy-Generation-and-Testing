#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volatility regime filter (ATR ratio) and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for volatility regime.
- Volatility regime: ATR(7)/ATR(30) > 1.5 indicates high volatility (breakout favorable), < 0.8 indicates low volatility (mean reversion).
- Entry: Long when price breaks above Donchian(20) upper AND vol regime > 1.5 (bullish breakout in high vol).
         Short when price breaks below Donchian(20) lower AND vol regime > 1.5 (bearish breakout in high vol).
         In low vol (regime < 0.8): Long when price touches Donchian lower AND reverses up (close > low).
                                  Short when price touches Donchian upper AND reverses down (close < high).
- Exit: Opposite Donchian breakout or volatility regime shift to opposite extreme.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ATR(7) and ATR(30) on 1d for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(np.diff(high_1d))
    tr2 = np.abs(high_1d[1:] - low_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(7) and ATR(30)
    atr7 = pd.Series(tr).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr30 = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Volatility regime: ATR(7)/ATR(30)
    vol_regime = np.where(atr30 > 0, atr7 / atr30, 1.0)
    
    # Align 1d volatility regime to 4h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for ATR and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_regime_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_reg = vol_regime_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if vol_reg > 1.5:  # High volatility regime: breakout strategy
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower Donchian
                    elif curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                elif vol_reg < 0.8:  # Low volatility regime: mean reversion at extremes
                    # Long when price touches lower Donchian and shows reversal (close > low)
                    if curr_low <= lowest_low[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Donchian and shows reversal (close < high)
                    elif curr_high >= highest_high[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR volatility regime shifts to low vol
            if curr_close < donchian_mid[i] or vol_reg < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR volatility regime shifts to low vol
            if curr_close > donchian_mid[i] or vol_reg < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0