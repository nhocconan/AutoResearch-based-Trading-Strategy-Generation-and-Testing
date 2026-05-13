#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA trend filter and volume confirmation. 
# Long when Williams %R < -80 (oversold) and price > 1d EMA50 (uptrend) with volume > 1.3x average.
# Short when Williams %R > -20 (overbought) and price < 1d EMA50 (downtrend) with volume > 1.3x average.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year on 6h timeframe.
# Williams %R identifies exhaustion points; EMA filter ensures trend alignment; volume confirms conviction.
# Works in bull markets via buying dips in uptrend and in bear markets via selling rallies in downtrend.

name = "6h_WilliamsR_MeanReversion_1dEMATrend_VolumeConfirm_v1"
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
    
    # Calculate Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe (wait for 1d bar to close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.3x average
            if (williams_r[i] < -80 and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.3x average
            elif (williams_r[i] > -20 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (exiting oversold) OR stoploss via signal=0
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (exiting overbought) OR stoploss via signal=0
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals