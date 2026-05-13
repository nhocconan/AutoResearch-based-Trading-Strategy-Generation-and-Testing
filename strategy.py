#/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA trend filter and volume confirmation capture directional moves in both bull and bear markets. The weekly EMA provides a robust trend filter that adapts to long-term market regimes, while Camarilla levels offer precise intraday support/resistance. Volume confirmation ensures breakouts have conviction. Designed for low trade frequency (10-25/year) with clear entry/exit rules to minimize fee drag.
"""

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Calculate daily Camarilla levels (using previous day's range)
    # R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first day's values to avoid NaN
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    R1 = prev_close + 1.1 * camarilla_range / 12
    S1 = prev_close - 1.1 * camarilla_range / 12
    
    # Calculate weekly EMA 50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and weekly uptrend
            if close[i] > R1[i] and volume_confirm[i]:
                # Additional filter: only take long if price above weekly EMA50 (uptrend filter)
                if close[i] > ema_50_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and weekly downtrend
            elif close[i] < S1[i] and volume_confirm[i]:
                # Additional filter: only take short if price below weekly EMA50 (downtrend filter)
                if close[i] < ema_50_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S1 (support break) or weekly trend turns down
            if close[i] < S1[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R1 (resistance break) or weekly trend turns up
            if close[i] > R1[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals