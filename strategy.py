#!/usr/bin/env python3
# Hypothesis: 12h Williams %R reversal with 1d trend filter and volume confirmation.
# Enters long when Williams %R crosses above -80 from below with 1d bullish trend (close > EMA34) and volume > 1.8x MA20.
# Enters short when Williams %R crosses below -20 from above with 1d bearish trend (close < EMA34) and volume > 1.8x MA20.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Williams %R captures overbought/oversold reversals that work in both bull and bear markets when combined with trend and volume filters.
# Targets 12-37 trades/year to avoid fee drag while capturing meaningful reversals.

name = "12h_WilliamsR_Reversal_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for Williams %R calculation (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Overbought: > -20, Oversold: < -80
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d data for trend filter (EMA34)
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.8x 20-period average (stricter to reduce trades)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from below with 1d bullish trend and volume spike
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                close[i] > ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from above with 1d bearish trend and volume spike
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  close[i] < ema34_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) or trend turns bearish
            if williams_r_aligned[i] > -20 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) or trend turns bullish
            if williams_r_aligned[i] < -80 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals