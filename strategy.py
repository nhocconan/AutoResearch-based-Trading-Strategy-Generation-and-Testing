# 4H_CAMARILLA_R1_S1_BREAKOUT_TREND_VOLUME
# Strategy type: Price channel breakout + trend filter + volume confirmation
# Timeframe: 4h (primary), 1d trend filter
# Hypothesis: Camarilla R1/S1 levels act as intraday support/resistance. 
# Breakouts above R1 or below S1 with volume confirmation and daily trend alignment 
# capture momentum moves. Works in bull/bear markets by only taking trades in 
# direction of daily trend, reducing false breakouts. Low trade frequency target 
# (20-40/year) minimizes fee drag.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_TREND_VOLUME"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for intraday trading"""
    # Pivot point
    pivot = (high + low + close) / 3.0
    # Range
    range_ = high - low
    # Camarilla levels
    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    s3 = close - (range_ * 1.1 / 4)
    s4 = close - (range_ * 1.1 / 2)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

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
    # Daily EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 4h data
    _, r1, _, _, _, s1, _, _, _ = calculate_camarilla(high, low, close)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and daily uptrend
            if (close[i] > r1[i] and close[i-1] <= r1[i-1] and 
                volume_confirm[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation and daily downtrend
            elif (close[i] < s1[i] and close[i-1] >= s1[i-1] and 
                  volume_confirm[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot level or breaks below S1
            if close[i] <= (r1[i] + s1[i])/2 or close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot level or breaks above R1
            if close[i] >= (r1[i] + s1[i])/2 or close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals