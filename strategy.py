#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 1d EMA trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND 1d EMA34 uptrend AND volume > 1.5 * volume MA(20)
- Short when price breaks below Camarilla S1 AND 1d EMA34 downtrend AND volume > 1.5 * volume MA(20)
- Uses Camarilla levels from prior daily bar (completed bar only) for structure-based breakouts
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike filter confirms institutional participation
- Designed for moderate frequency (target 20-50 trades/year) to minimize fee drag
- Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend reversal
- Focus on BTC/ETH as primary symbols, SOL as secondary
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior daily bar
    # Camarilla R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    camarilla_r1 = daily_close + 1.1 * (daily_high - daily_low) / 12
    camarilla_s1 = daily_close - 1.1 * (daily_high - daily_low) / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed daily bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(daily_close).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Volume spike filter: volume > 1.5 * volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND daily uptrend AND volume spike
            if close[i] > camarilla_r1_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND daily downtrend AND volume spike
            elif close[i] < camarilla_s1_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR daily trend turns down
            if close[i] < camarilla_s1_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR daily trend turns up
            if close[i] > camarilla_r1_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0