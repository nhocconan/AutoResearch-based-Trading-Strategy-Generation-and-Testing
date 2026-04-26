#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrendFilter_VolumeConfirm_v1
Hypothesis: 1d Camarilla pivot breakout strategy with weekly trend filter and volume confirmation.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Camarilla pivot levels (R1, S1) calculated from prior 1d candle
- Weekly EMA200 filter ensures trades align with higher timeframe trend
- Volume confirmation: current volume > 1.5x 20-day average volume
- Long when price breaks above R1 AND weekly uptrend AND volume confirmation
- Short when price breaks below S1 AND weekly downtrend AND volume confirmation
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the weekly trend and using Camarilla for precise entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need enough data for volume average
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            i == 0):  # Need prior day for Camarilla calculation
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla pivot levels from prior day
        # Based on prior day's high, low, close
        prior_high = high[i-1]
        prior_low = low[i-1]
        prior_close = close[i-1]
        
        # Calculate pivot point
        pivot = (prior_high + prior_low + prior_close) / 3
        
        # Calculate Camarilla levels
        range_ = prior_high - prior_low
        r1 = pivot + (range_ * 1.1 / 12)
        s1 = pivot - (range_ * 1.1 / 12)
        
        # Volume confirmation: current volume > 1.5x 20-day average
        volume_confirmed = volume[i] > (1.5 * vol_ma_20[i])
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND weekly uptrend AND volume confirmation
            if close[i] > r1 and weekly_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND weekly downtrend AND volume confirmation
            elif close[i] < s1 and weekly_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 (reversion to mean) OR weekly trend turns down
            if close[i] < s1 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 (reversion to mean) OR weekly trend turns up
            if close[i] > r1 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrendFilter_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0