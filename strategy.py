#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 12h EMA50 trend filter and volume spike confirmation.
# Long when Williams %R(14) < -80 (oversold) AND 12h EMA50 is rising (uptrend) AND volume > 2.0x 20-period average.
# Short when Williams %R(14) > -20 (overbought) AND 12h EMA50 is falling (downtrend) AND volume > 2.0x 20-period average.
# Williams %R identifies extreme short-term reversals that work in both bull and bear markets.
# 12h EMA50 ensures we trade with the intermediate-term trend, reducing counter-trend whipsaws.
# Volume spike (>2.0x average) confirms institutional participation in the reversal.
# Target: 75-150 total trades over 4 years (19-38/year) on 6h timeframe.

name = "6h_WilliamsR_MeanReversion_12hEMA50_Trend_Volume_v1"
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
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i-1]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND 12h EMA50 rising (trending up) AND volume spike
            if williams_r[i] < -80 and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND 12h EMA50 falling (trending down) AND volume spike
            elif williams_r[i] > -20 and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (exiting oversold territory) OR volume dies
            if williams_r[i] > -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (exiting overbought territory) OR volume dies
            if williams_r[i] < -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals