#!/usr/bin/env python3
# Hypothesis: 4h Williams %R mean reversion with 1d EMA200 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) and price > 1d EMA200 (uptrend) and volume > 1.5x average.
# Short when Williams %R > -20 (overbought) and price < 1d EMA200 (downtrend) and volume > 1.5x average.
# Exit when Williams %R crosses above -50 for longs or below -50 for shorts.
# Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years on 4h timeframe.
# Williams %R identifies exhaustion points; EMA200 ensures trend alignment; volume confirms momentum.
# This combination avoids overtrading by requiring multiple confluence factors while capturing mean reversion within trends.

name = "4h_WilliamsR_MeanReversion_1dEMA200_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    period = 14
    if n < period:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on 1d data
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Align 1d EMA200 to 4h timeframe (wait for 1d bar to close)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(period, 20) + 1, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) and price > 1d EMA200 (uptrend) and volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_200_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) and price < 1d EMA200 (downtrend) and volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (recovering from oversold)
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (declining from overbought)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals