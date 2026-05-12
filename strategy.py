#!/usr/bin/env python3
"""
6h_12h_1d_AdaptiveChannel_Breakout_Target
Hypothesis: Adaptive channel breakouts using 6h Donchian (20) with 12h trend filter and volume confirmation.
Targets 6h timeframe to reduce trade frequency (target: 15-35 trades/year) while using proven breakout structure.
Uses dynamic channel width based on ATR volatility to adapt to changing market conditions.
Only takes long when price breaks above upper channel with volume spike and 12h uptrend, short when breaks below lower channel with volume spike and 12h downtrend.
Adaptive channel prevents overtrading in low volatility and undertrading in high volatility.
"""

name = "6h_12h_1d_AdaptiveChannel_Breakout_Target"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Adaptive channel: Donchian with ATR-based width adjustment
    # Calculate ATR(14) for volatility measurement
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Base Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Adaptive channel width: base width + ATR multiplier
    base_width = donch_high - donch_low
    atr_multiplier = 0.5  # Adjusts channel width based on volatility
    channel_width = base_width + (atr_multiplier * atr)
    
    # Upper and lower adaptive channels
    upper_channel = donch_high + (channel_width * 0.1)  # 10% of adaptive width above Donchian high
    lower_channel = donch_low - (channel_width * 0.1)   # 10% of adaptive width below Donchian low
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper channel + volume spike + price above 12h EMA50
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower channel + volume spike + price below 12h EMA50
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below upper channel OR closes below 12h EMA50
            if close[i] < upper_channel[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above lower channel OR closes above 12h EMA50
            if close[i] > lower_channel[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals