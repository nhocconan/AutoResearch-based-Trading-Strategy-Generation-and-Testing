#!/usr/bin/env python3
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
    
    # Load 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on 1-day timeframe
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First value
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate ATR(14) on 6-hour timeframe
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = tr_6h_1[0]
    tr_6h_3[0] = tr_6h_1[0]
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    
    atr_6h = np.zeros_like(tr_6h)
    atr_6h[atr_period-1] = np.mean(tr_6h[:atr_period])
    for i in range(atr_period, len(tr_6h)):
        atr_6h[i] = (atr_6h[i-1] * (atr_period-1) + tr_6h[i]) / atr_period
    
    # Calculate ATR ratio: 6h ATR / daily ATR (volatility regime filter)
    atr_ratio_raw = atr_6h / atr
    
    # Align ATR ratio to 6-hour timeframe
    atr_ratio = align_htf_to_ltf(prices, df_1d, atr_ratio_raw)
    
    # Volume spike filter (20-period on 6h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20  # Require 1.5x volume for confirmation
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready or outside session
        if (np.isnan(atr_ratio[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Low volatility regime: ATR ratio < 0.8 (6h vol < 80% of daily vol)
            # Look for breakouts with volume confirmation
            if atr_ratio[i] < 0.8:
                # Donchian breakout (20-period) with volume
                if i >= 20:
                    highest_20 = np.max(high[i-20:i])
                    lowest_20 = np.min(low[i-20:i])
                    
                    if close[i] > highest_20 and vol_spike[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < lowest_20 and vol_spike[i]:
                        signals[i] = -0.25
                        position = -1
        else:
            # Exit conditions: volatility expansion or mean reversion
            if position == 1:  # Long position
                # Exit if volatility expands (ATR ratio > 1.2) or price drops below entry - 1*ATR
                if atr_ratio[i] > 1.2 or close[i] < close[i-1] - atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Short position
                # Exit if volatility expands or price rises above entry + 1*ATR
                if atr_ratio[i] > 1.2 or close[i] > close[i-1] + atr_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ATR_Ratio_Vol_Spike_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0