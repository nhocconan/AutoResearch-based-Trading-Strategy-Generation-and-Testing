# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as key support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and 12h EMA trend filter capture sustained moves.
# Designed for low trade frequency (20-40/year) to work in both bull and bear markets by avoiding chop.
# Uses 12h EMA for trend filter to reduce whipsaw and improve trend following.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels: R1-R4, S1-S4"""
    range_ = high - low
    close = close
    r1 = close + range_ * 1.1 / 12
    s1 = close - range_ * 1.1 / 12
    r2 = close + range_ * 1.1 / 6
    s2 = close - range_ * 1.1 / 6
    r3 = close + range_ * 1.1 / 4
    s3 = close - range_ * 1.1 / 4
    r4 = close + range_ * 1.1 / 2
    s4 = close - range_ * 1.1 / 2
    return r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot points
    df_daily = get_htf_data(prices, '1d')
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate daily Camarilla levels
    r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
        daily_high, daily_low, daily_close
    )
    
    # Align daily Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_daily, r1)
    s1_4h = align_htf_to_ltf(prices, df_daily, s1)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Break above R1 with volume confirmation and uptrend (price > 12h EMA)
            if close[i] > r1_4h[i] and close[i-1] <= r1_4h[i-1] and volume_confirm[i] and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume confirmation and downtrend (price < 12h EMA)
            elif close[i] < s1_4h[i] and close[i-1] >= s1_4h[i-1] and volume_confirm[i] and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 (stop) or reaches R2 (take profit)
            if close[i] < s1_4h[i] or close[i] >= r2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 (stop) or reaches S2 (take profit)
            if close[i] > r1_4h[i] or close[i] <= s2_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Note: r2_4h and s2_4h are derived from r2 and s2 alignment (added below)
    # Calculate R2 and S2 for exit levels
    r2_4h = align_htf_to_ltf(prices, df_daily, r2)
    s2_4h = align_htf_to_ltf(prices, df_daily, s2)