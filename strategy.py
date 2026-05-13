#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume spike.
# Enters long when Williams %R(14) crosses above -80 from below with 1d bullish trend (close > EMA34) and volume > 2.0x MA20.
# Enters short when Williams %R(14) crosses below -20 from above with 1d bearish trend (close < EMA34) and volume > 2.0x MA20.
# Exits when Williams %R crosses above -20 for longs or below -80 for shorts.
# Uses discrete position sizing (0.25) to minimize fee drag.
# Designed for low trade frequency (~12-37/year) by requiring confluence of reversal signal, trend alignment, and volume confirmation.
# Works in both bull and bear markets: 1d trend filter ensures alignment with higher timeframe direction,
# while Williams %R provides timely reversal signals in ranging markets and volume confirmation reduces false signals.

name = "6h_WilliamsR_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Williams %R on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero or near-zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(williams_r[i]) or \
           np.isnan(vol_ma20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below with 1d bullish trend and volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above with 1d bearish trend and volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought)
            if williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold)
            if williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals