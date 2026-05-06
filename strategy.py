#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Elder Ray (Bull/Bear Power) with 1d EMA trend filter and volume confirmation
# Long when Bull Power > 0, price > 1d EMA200, and volume > 1.2x 20-period average
# Short when Bear Power < 0, price < 1d EMA200, and volume > 1.2x 20-period average
# Elder Ray identifies bull/bear power via EMA13, trend filter avoids counter-trend trades, volume confirms conviction
# Designed to work in bull markets via sustained Bull Power and in bear markets via sustained Bear Power
# Target: 50-150 total trades over 4 years = 12-37/year with 0.25 position sizing

name = "6h_ElderRay_EMA200_Volume_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA13 for Elder Ray
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_12h[-len(high):]  # Align to same length as prices
    bear_power = low - ema13_12h[-len(low):]
    
    # Calculate 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: >1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power positive, price above EMA200, volume confirmation
            if bull_power[i] > 0 and close[i] > ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, price below EMA200, volume confirmation
            elif bear_power[i] < 0 and close[i] < ema200_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bear Power turns negative (trend change)
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power turns positive (trend change)
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals