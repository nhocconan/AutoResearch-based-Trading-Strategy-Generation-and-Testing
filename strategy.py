# 4h_GoldenCross_Reversal_With_Volume_And_Trend
# Hypothesis: In volatile crypto markets, reversals at key moving average crossovers
# with volume confirmation and higher timeframe trend alignment provide high-probability
# entries. This strategy uses EMA50/EMA200 golden/death cross on 4h as the primary signal,
# confirmed by volume spike and aligned with 1d trend (EMA50) to avoid counter-trend trades.
# Works in bull markets (buying golden crosses in uptrend) and bear markets (selling death crosses in downtrend).
# Targets 20-40 trades/year to minimize fee drag.

name = "4h_GoldenCross_Reversal_With_Volume_And_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 on 4h chart for golden/death cross
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = close_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period MA on 4h chart = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA200 (200), EMA50 (50), EMA50_1d (50), volume MA (20)
    start_idx = max(200, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter: uptrend if price > EMA50_1d, downtrend if price < EMA50_1d
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Golden cross: EMA50 crosses above EMA200
        # Death cross: EMA50 crosses below EMA200
        if i > 0:
            golden_cross = (ema_50[i] > ema_200[i]) and (ema_50[i-1] <= ema_200[i-1])
            death_cross = (ema_50[i] < ema_200[i]) and (ema_50[i-1] >= ema_200[i-1])
        else:
            golden_cross = False
            death_cross = False
        
        if position == 0:
            # Long entry: golden cross + uptrend + volume confirmation
            if golden_cross and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: death cross + downtrend + volume confirmation
            elif death_cross and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: death cross or trend breakdown
            if death_cross or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: golden cross or trend reversal
            if golden_cross or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals