#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R4 AND 1d close > EMA34 (uptrend) AND volume spike.
# Short when price breaks below S4 AND 1d close < EMA34 (downtrend) AND volume spike.
# Uses Camarilla levels from 1d for structure, 1d EMA34 for trend, and volume for momentum confirmation.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drag in bear markets.
# Works in bull markets via R4 breakouts in uptrend and in bear markets via S4 breakdowns in downtrend.
name = "6h_Camarilla_R4S4_Breakout_1dTrend_Volume"
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
    
    # Load 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1 / 2)
    # S4 = C - (Range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h (wait for 1d close)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d trend: EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    trend_up = close_1d > ema_34_1d
    trend_down = close_1d < ema_34_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: current volume > 2.0 * 24-period EMA (6h: 24 periods = 4 days)
    vol_ema_24 = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ema_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 AND 1d uptrend AND volume spike
            long_condition = (close[i] > r4_1d_aligned[i]) and trend_up_aligned[i] and volume_spike[i]
            # Short: price breaks below S4 AND 1d downtrend AND volume spike
            short_condition = (close[i] < s4_1d_aligned[i]) and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price re-enters below R4 OR 1d trend turns down
            if (close[i] < r4_1d_aligned[i]) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price re-enters above S4 OR 1d trend turns up
            if (close[i] > s4_1d_aligned[i]) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals