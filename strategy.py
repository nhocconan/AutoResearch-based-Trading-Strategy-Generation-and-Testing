#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 13-period EMA for Elder Ray (standard setting)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = ema13_1d - df_1d['low'].values
    
    # 1d trend filter: price above/below EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = (close_1d > ema34_1d).astype(float)
    trend_down = (close_1d < ema34_1d).astype(float)
    
    # Align Elder Ray and trend to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    trend_up_6h = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_6h = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(trend_up_6h[i]) or np.isnan(trend_down_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power positive with volume spike and 1d uptrend
            long_cond = (bull_power_6h[i] > 0 and vol_spike[i] and trend_up_6h[i] > 0.5)
            
            # Short entry: Bear Power positive with volume spike and 1d downtrend
            short_cond = (bear_power_6h[i] > 0 and vol_spike[i] and trend_down_6h[i] > 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative (momentum fade)
            if bull_power_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns negative (momentum fade)
            if bear_power_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray (Bull/Bear Power) measures buying/selling pressure relative to 13-period EMA.
# Combines with 1d EMA34 trend filter and volume spike confirmation for institutional-grade entries.
# Works in bull markets (Bull Power + uptrend) and bear markets (Bear Power + downtrend).
# Conservative 0.25 position size limits drawdown; exits on momentum fade prevent whipsaws.
# Target: 15-35 trades/year to avoid fee drag while capturing strong institutional moves.