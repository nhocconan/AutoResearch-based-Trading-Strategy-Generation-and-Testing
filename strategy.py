#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation (>2.0x 30-period average), and ATR(14) trailing stoploss.
- Donchian(20) provides clear structure for breakouts in both bull and bear markets.
- 1d EMA34 ensures alignment with higher-timeframe trend to reduce counter-trend trades.
- Volume spike confirms breakout validity and reduces false signals.
- ATR-based trailing stoploss manages risk and adapts to volatility.
- Discrete position sizing (0.30) balances return potential with fee minimization.
- Target trades: 75-200 total over 4 years (19-50/year) on 4h timeframe to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR(14) for volatility and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 30-period average volume
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_long = 0.0  # Track highest high since long entry
    lowest_since_short = 0.0  # Track lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(30, 34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
                lowest_since_short = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume spike and above 1d EMA34 (bullish higher-timeframe trend)
            if close[i] > highest_high[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
                highest_since_long = high[i]
            # Short: break below Donchian low with volume spike and below 1d EMA34 (bearish higher-timeframe trend)
            elif close[i] < lowest_low[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
                lowest_since_short = low[i]
        elif position == 1:
            # Update highest high since long entry
            highest_since_long = max(highest_since_long, high[i])
            # Long exit: price closes below highest_high - 2.5 * ATR (trailing stop) OR below Donchian low
            trailing_stop_long = highest_since_long - 2.5 * atr[i]
            if close[i] < trailing_stop_long or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                highest_since_long = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since short entry
            lowest_since_short = min(lowest_since_short, low[i])
            # Short exit: price closes above lowest_low + 2.5 * ATR (trailing stop) OR above Donchian high
            trailing_stop_short = lowest_since_short + 2.5 * atr[i]
            if close[i] > trailing_stop_short or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                lowest_since_short = 0.0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA34_VolumeSpike_ATRTrailingStop_v1"
timeframe = "4h"
leverage = 1.0