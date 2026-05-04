#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla Pivot Breakout + 4h Trend + Volume Spike
# Camarilla levels (R1/S1) provide tight intraday support/resistance for precise entries.
# 4h EMA50 filters trend direction to avoid counter-trend whipsaws.
# Volume confirmation ensures momentum behind breakouts.
# Designed for 15-37 trades/year on 1h to minimize fee drag in BTC/ETH.
# Works in bull markets via long R1 breakouts in uptrend and bear markets via short S1 breakdowns in downtrend.

name = "1h_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h Camarilla pivot points (R1, S1) using previous bar's OHLC
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Pivot = (high + low + close) / 3
    pivot = typical_price
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    r1 = close + 1.1 * (high - low) / 12
    s1 = close - 1.1 * (high - low) / 12
    
    # Shift to get previous bar's levels (no look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r1_prev[i]) or np.isnan(s1_prev[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1 AND 4h uptrend AND volume spike
            if (close[i] > r1_prev[i] and  # Break above R1
                close[i] > ema_50_aligned[i] and  # 4h uptrend
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S1 AND 4h downtrend AND volume spike
            elif (close[i] < s1_prev[i] and  # Break below S1
                  close[i] < ema_50_aligned[i] and  # 4h downtrend
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below pivot OR 4h trend turns down
            if close[i] < pivot[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above pivot OR 4h trend turns up
            if close[i] > pivot[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals