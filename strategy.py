#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (EMA50) and volume confirmation
# Long when price breaks above R1 AND close > 4h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below S1 AND close < 4h EMA50 AND volume > 1.5x 20-period average
# Uses discrete position size 0.20 to minimize fee churn. Target: 15-37 trades/year by requiring confluence of 4h trend, 1h breakout, and volume spike.
# Works in bull markets via breakouts with trend, in bear markets via breakdowns with trend filter.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate previous day's high, low, close for Camarilla levels
    # Use 1d data to get proper daily OHLC
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 levels for each 1d bar
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1_1d = close_1d + 1.1 * camarilla_range / 12
    s1_1d = close_1d - 1.1 * camarilla_range / 12
    
    # Al Camarilla levels to 1h timeframe (wait for daily close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or \
           np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R1 AND price > 4h EMA50 AND volume filter
            if close[i] > r1_1d_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S1 AND price < 4h EMA50 AND volume filter
            elif close[i] < s1_1d_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 OR price < 4h EMA50 (trend break)
            if close[i] < s1_1d_aligned[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above R1 OR price > 4h EMA50 (trend break)
            if close[i] > r1_1d_aligned[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals