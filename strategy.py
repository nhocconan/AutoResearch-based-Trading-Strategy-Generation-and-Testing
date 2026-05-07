#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (bull/bear power) with 1d trend filter and volume confirmation.
# Long when bull power crosses above 0 (bullish momentum) AND 1d close > EMA50 (uptrend) AND volume spike.
# Short when bear power crosses below 0 (bearish momentum) AND 1d close < EMA50 (downtrend) AND volume spike.
# Elder Ray captures bull/bear power via EMA13, providing early momentum signals.
# Combined with 1d trend filter to avoid counter-trend trades and volume to confirm strength.
# Designed for moderate trade frequency (target: 20-40/year) with strong risk control.
name = "6h_ElderRay_1dTrend_Volume"
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
    
    # Elder Ray Index (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Elder Ray signals: bull power crosses above 0 (long), bear power crosses below 0 (short)
    bull_power_cross_up = (bull_power > 0) & (np.roll(bull_power, 1) <= 0)
    bear_power_cross_down = (bear_power < 0) & (np.roll(bear_power, 1) >= 0)
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up = close_1d > ema_50_1d
    trend_down = close_1d < ema_50_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Elder Ray and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bull power crosses above 0 + 1d uptrend + volume spike
            long_condition = bull_power_cross_up[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: bear power crosses below 0 + 1d downtrend + volume spike
            short_condition = bear_power_cross_down[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bull power crosses below 0 (momentum loss) or 1d trend turns down
            if (bull_power[i] < 0 and np.roll(bull_power, 1)[i] >= 0) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bear power crosses above 0 (momentum loss) or 1d trend turns up
            if (bear_power[i] > 0 and np.roll(bear_power, 1)[i] <= 0) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals