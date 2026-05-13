#!/usr/bin/env python3
# Hypothesis: 6h Williams %R overbought/oversold with 12h EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 12h EMA50 (uptrend) AND volume > 1.5x average.
# Short when Williams %R > -20 (overbought) AND price < 12h EMA50 (downtrend) AND volume > 1.5x average.
# Exit on opposite Williams %R level (%R > -50 for long exit, %R < -50 for short exit) or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Williams %R captures mean reversion in ranges while trend filter ensures we trade with the 12h momentum.
# Volume spike confirms conviction behind the move. Works in bull markets via dips in uptrend and in bear markets via rallies in downtrend.

name = "6h_WilliamsR_12hTrend_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Williams %R(14) on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: oversold (%R < -80) + uptrend (price > 12h EMA50) + volume spike
            if williams_r[i] < -80 and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: overbought (%R > -20) + downtrend (price < 12h EMA50) + volume spike
            elif williams_r[i] > -20 and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: %R > -50 (leaving oversold territory) OR trend reversal (price < 12h EMA50)
            if williams_r[i] > -50 or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: %R < -50 (leaving overbought territory) OR trend reversal (price > 12h EMA50)
            if williams_r[i] < -50 or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals