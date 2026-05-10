#!/usr/bin/env python3
# 4h_Elder_Ray_1dTrend_Volume_Spike
# Hypothesis: Elder Ray Index (bull power = high - EMA13, bear power = low - EMA13) with 1d EMA trend filter and volume spike.
# Bull power > 0 and rising indicates bullish momentum; bear power < 0 and falling indicates bearish momentum.
# Works in bull/bear markets by aligning with daily trend direction. Targets 20-50 trades/year to minimize fee drag.
# Uses 13-period EMA for consistency with Elder Ray standard settings.

name = "4h_Elder_Ray_1dTrend_Volume_Spike"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 34, 20)  # Warmup for EMA13, daily EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Elder Ray signals: rising bull power = bullish, falling bear power = bearish
        bullish_signal = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        bearish_signal = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long entry: bullish signal + uptrend + volume spike
            if bullish_signal and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish signal + downtrend + volume spike
            elif bearish_signal and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bull power turns negative or trend reversal
            if bull_power[i] <= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bear power turns positive or trend reversal
            if bear_power[i] >= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals