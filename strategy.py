#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) on 12h with 1d trend filter and volume spike confirmation.
# In bull markets, buy near S3 with stop below S4; in bear markets, sell near R3 with stop above R4.
# Uses 1d EMA trend filter to align with higher timeframe direction and volume confirmation to avoid false breakouts.
# Targets 15-25 trades/year to minimize fee drag and works in both bull/bear regimes.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for 12h timeframe
    # Using previous bar's high, low, close
    phigh = np.concatenate([[high[0]], high[:-1]])
    plow = np.concatenate([[low[0]], low[:-1]])
    pclose = np.concatenate([[close[0]], close[:-1]])
    
    # Camarilla calculations
    R3 = pclose + (phigh - plow) * 1.1 / 4
    S3 = pclose - (phigh - plow) * 1.1 / 4
    R4 = pclose + (phigh - plow) * 1.1 / 2
    S4 = pclose - (phigh - plow) * 1.1 / 2
    
    # Get daily EMA for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period MA for 12h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34)  # Warmup for volume MA and daily EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(R4[i]) or np.isnan(S4[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price crosses above S3 with uptrend and volume spike
            if close[i] > S3[i] and close[i-1] <= S3[i-1] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses below R3 with downtrend and volume spike
            elif close[i] < R3[i] and close[i-1] >= R3[i-1] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S4 or trend reversal
            if close[i] < S4[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R4 or trend reversal
            if close[i] > R4[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals