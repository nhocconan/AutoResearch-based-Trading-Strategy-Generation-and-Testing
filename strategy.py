#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR-based trailing stop.
Long when price breaks above Donchian upper band AND price > 1d EMA50 AND volume > 1.5x 20-period MA.
Short when price breaks below Donchian lower band AND price < 1d EMA50 AND volume > 1.5x 20-period MA.
Exit when trailing ATR stop is hit (3x ATR from extreme) or opposite breakout occurs.
Designed for 4h timeframe to target 20-50 trades/year with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = 0.0  # for long trailing stop
    lowest_low = 0.0    # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # need EMA50, Donchian20, ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high = 0.0
                lowest_low = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > 1d EMA50 AND volume spike
            if close[i] > donchian_upper[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high = high[i]
            # Short: price breaks below Donchian lower AND price < 1d EMA50 AND volume spike
            elif close[i] < donchian_lower[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low = low[i]
        else:
            # Update trailing extremes
            if position == 1:
                highest_high = max(highest_high, high[i])
                # Exit long when price drops 3*ATR from highest high
                if close[i] < highest_high - 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high = 0.0
                # Also exit on opposite Donchian break
                elif close[i] < donchian_lower[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                lowest_low = min(lowest_low, low[i])
                # Exit short when price rises 3*ATR from lowest low
                if close[i] > lowest_low + 3.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low = 0.0
                # Also exit on opposite Donchian break
                elif close[i] > donchian_upper[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low = 0.0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0