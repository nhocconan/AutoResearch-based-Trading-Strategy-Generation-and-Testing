#!/usr/bin/env python3
# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation.
# Enters long when Williams %R crosses above -80 from oversold with 1d bullish trend (close > EMA50) and volume > 1.5x MA20.
# Enters short when Williams %R crosses below -20 from overbought with 1d bearish trend (close < EMA50) and volume > 1.5x MA20.
# Exits when Williams %R returns to the opposite extreme (-20 for long, -80 for short) or volume drops below average.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Designed for low trade frequency (~12-37/year) to work in both bull and bear markets by requiring mean reversion extremes with trend alignment.

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
    
    # Get 1d data for Williams %R calculation (14-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d data for trend filter (EMA50)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 from oversold with 1d bullish trend and volume spike
            if i > 0 and williams_r_aligned[i-1] <= -80 and williams_r_aligned[i] > -80 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 from overbought with 1d bearish trend and volume spike
            elif i > 0 and williams_r_aligned[i-1] >= -20 and williams_r_aligned[i] < -20 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -20 or volume drops below average
            if williams_r_aligned[i] >= -20 or volume[i] <= vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -80 or volume drops below average
            if williams_r_aligned[i] <= -80 or volume[i] <= vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals