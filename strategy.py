#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume spike confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong bullish signal when Bull Power > 0 AND Bear Power < 0 (price above EMA13 with strong upward momentum)
# Strong bearish signal when Bull Power < 0 AND Bear Power > 0 (price below EMA13 with strong downward momentum)
# 12h EMA50 filter ensures we only trade in the direction of the higher timeframe trend
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in bull markets via long signals when price > EMA13 with upward momentum
# Works in bear markets via short signals when price < EMA13 with downward momentum
# Discrete sizing 0.25 minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_BullBearPower_12hEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA13 calculation (MTF Rule #1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components: Bull Power and Bear Power
    bull_power = high - ema_13_1d_aligned  # High - EMA13
    bear_power = low - ema_13_1d_aligned   # Low - EMA13
    
    # Load 12h data ONCE before loop for trend filter (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 13, 50, 20)  # warmup for EMA13, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if curr_volume_spike:
                # Bullish entry: Bull Power > 0 AND Bear Power < 0 (strong upward momentum) AND price above 12h EMA50 (uptrend)
                if curr_bull_power > 0 and curr_bear_power < 0 and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Bull Power < 0 AND Bear Power > 0 (strong downward momentum) AND price below 12h EMA50 (downtrend)
                elif curr_bull_power < 0 and curr_bear_power > 0 and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when bullish momentum weakens (Bull Power <= 0)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bearish momentum weakens (Bear Power <= 0)
            if curr_bear_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals