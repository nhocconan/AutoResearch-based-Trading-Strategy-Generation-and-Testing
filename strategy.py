# 6h_Camarilla_Pivot_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla R3/S3 breakout with weekly trend filter and volume confirmation.
# In bull markets: break above R3 with weekly uptrend and volume spike = long.
# In bear markets: break below S3 with weekly downtrend and volume spike = short.
# Uses Camarilla levels from 1d data, weekly trend from weekly data.
# Target: 15-35 trades/year per symbol (60-140 total over 4 years).
# Weekly trend filter avoids counter-trend trades in strong trends.

name = "6h_Camarilla_Pivot_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    R4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    S4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1w = df_1w['close'].values > ema20_1w
    downtrend_1w = df_1w['close'].values < ema20_1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w)
    downtrend_1w_aligned = align_htf_to_ltf(prices, df_1w, downtrend_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_conf = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get current values
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        r4 = R4_aligned[i]
        s4 = S4_aligned[i]
        uptrend = uptrend_1w_aligned[i]
        downtrend = downtrend_1w_aligned[i]
        vol_conf = volume_conf[i]
        
        if position == 0:
            # LONG: break above R3 with weekly uptrend and volume confirmation
            if close[i] > r3 and uptrend and vol_conf:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S3 with weekly downtrend and volume confirmation
            elif close[i] < s3 and downtrend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: break below S4 (strong reversal) or weekly trend turns down
            if close[i] < s4 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: break above R4 (strong reversal) or weekly trend turns up
            if close[i] > r4 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals