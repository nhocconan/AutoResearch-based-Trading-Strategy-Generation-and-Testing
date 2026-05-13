#!/usr/bin/env python3
# Hypothesis: 12h Williams %R with 1d EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: long when %R crosses above -80 from below,
# short when %R crosses below -20 from above, only in direction of 1d EMA50 trend.
# Volume confirmation requires current volume > 1.5 * 20-period average volume.
# Designed for 12-37 trades/year by requiring strict alignment of momentum, trend, and volume.
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend.

name = "12h_WilliamsR_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams %R(14) = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(volume_confirm[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA50 (uptrend) AND volume confirmation
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) OR price < 1d EMA50 (trend break)
            if williams_r[i] >= -20 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) OR price > 1d EMA50 (trend break)
            if williams_r[i] <= -80 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals