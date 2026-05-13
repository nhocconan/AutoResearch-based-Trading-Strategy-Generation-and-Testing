#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily chart act as key intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation and daily trend filter capture
trend moves in both bull and bear markets. Designed for low trade frequency (15-30/year)
with clear entry/exit rules to minimize fee churn.
"""

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate previous day's Camarilla pivot levels (R1, S1)
    # Using daily OHLC from previous day
    prev_day_high = np.roll(high, 1)
    prev_day_low = np.roll(low, 1)
    prev_day_close = np.roll(close, 1)
    prev_day_high[0] = high[0]  # first bar uses current day's high as placeholder
    prev_day_low[0] = low[0]
    prev_day_close[0] = close[0]
    
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    r1 = pivot + (prev_day_high - prev_day_low) * 1.1 / 12
    s1 = pivot - (prev_day_high - prev_day_low) * 1.1 / 12
    
    # Daily trend filter: EMA 34
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and daily uptrend
            if close[i] > r1[i] and volume_confirm[i]:
                if close[i] > ema_34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and daily downtrend
            elif close[i] < s1[i] and volume_confirm[i]:
                if close[i] < ema_34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (reversal) or volume dries up
            if close[i] < s1[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (reversal) or volume dries up
            if close[i] > r1[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals