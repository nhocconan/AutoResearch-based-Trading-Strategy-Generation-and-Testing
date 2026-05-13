# 12h_1D_Camarilla_R1_S1_Breakout_With_Volume
# Hypothesis: Camarilla pivot levels from daily chart provide strong support/resistance
# levels. Price breaking above R1 or below S1 with volume confirmation indicates
# momentum continuation. In ranging markets, price tends to revert from extreme
# levels (R4/S4). This strategy combines breakout and mean-reversion logic with
# volume filtering to reduce false signals. Designed for low trade frequency
# (15-25/year) to work in both bull and bear markets by capturing breakouts
# in trends and reversals at extremes.

name = "12h_1D_Camarilla_R1_S1_Breakout_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for a given period"""
    # Pivot point
    pivot = (high + low + close) / 3.0
    # Range
    range_ = high - low
    # Camarilla levels
    r1 = close + range_ * 1.1 / 12
    r2 = close + range_ * 1.1 / 6
    r3 = close + range_ * 1.1 / 4
    r4 = close + range_ * 1.1 / 2
    s1 = close - range_ * 1.1 / 12
    s2 = close - range_ * 1.1 / 6
    s3 = close - range_ * 1.1 / 4
    s4 = close - range_ * 1.1 / 2
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate daily Camarilla levels
    _, r1, _, _, r4, s1, _, _, s4 = calculate_camarilla(
        daily_high, daily_low, daily_close
    )
    
    # Align daily Camarilla levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_daily, r1)
    r4_12h = align_htf_to_ltf(prices, df_daily, r4)
    s1_12h = align_htf_to_ltf(prices, df_daily, s1)
    s4_12h = align_htf_to_ltf(prices, df_daily, s4)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # MEAN REVERSION LONG: Price at S4 with volume confirmation
            if close[i] <= s4_12h[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # MEAN REVERSION SHORT: Price at R4 with volume confirmation
            elif close[i] >= r4_12h[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            # BREAKOUT LONG: Price breaks above R1 with volume confirmation
            elif close[i] > r1_12h[i] and close[i-1] <= r1_12h[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # BREAKOUT SHORT: Price breaks below S1 with volume confirmation
            elif close[i] < s1_12h[i] and close[i-1] >= s1_12h[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 (take profit) or breaks below S1 (stop)
            if close[i] >= r1_12h[i] or close[i] < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 (take profit) or breaks above R1 (stop)
            if close[i] <= s1_12h[i] or close[i] > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals