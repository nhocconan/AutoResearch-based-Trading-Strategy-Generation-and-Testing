#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2
Hypothesis: 1d Camarilla R1/S1 breakout with volume spike and weekly EMA20 trend filter.
- Long when price breaks above Camarilla R1 level (from prior 1d range) AND volume spike AND weekly EMA20 uptrend
- Short when price breaks below Camarilla S1 level AND volume spike AND weekly EMA20 downtrend
- Uses prior 1d range for Camarilla levels (structure-based edge from prior completed daily bar)
- Volume spike confirms institutional participation (2.0x 20-period average on 1d)
- Weekly EMA20 filter ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
- Designed for low frequency (target 7-25 trades/year) to minimize fee drag and improve test generalization
- Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend reversal
- Novelty: Tight Camarilla R1/S1 levels + weekly HTF trend filter on 1d timeframe for BTC/ETH edge in both bull/bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (completed bar only)
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    prior_1d_high = np.roll(df_1d['high'].values, 1)
    prior_1d_low = np.roll(df_1d['low'].values, 1)
    prior_1d_close = np.roll(df_1d['close'].values, 1)
    # First value is invalid due to roll
    prior_1d_high[0] = np.nan
    prior_1d_low[0] = np.nan
    prior_1d_close[0] = np.nan
    
    cam_r1 = prior_1d_close + (prior_1d_high - prior_1d_low) * 1.1 / 12
    cam_s1 = prior_1d_close - (prior_1d_high - prior_1d_low) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed for structure)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA20 for trend filter (needs completed weekly candle)
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    # Trend: 1 = uptrend (close > EMA20), -1 = downtrend (close < EMA20), 0 = neutral/invalid
    trend_1w = np.where(ema_20_1w_aligned > 0, 
                        np.where(close > ema_20_1w_aligned, 1, -1), 
                        0)
    
    # Calculate volume spike (20-period volume average on 1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for prior 1d, 20 for weekly EMA)
    start_idx = max(20, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trend_1w[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R1/S1 breakout conditions with volume confirmation and weekly trend filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND volume spike AND weekly uptrend
            if close[i] > cam_r1_aligned[i] and volume_spike[i] and trend_1w[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND volume spike AND weekly downtrend
            elif close[i] < cam_s1_aligned[i] and volume_spike[i] and trend_1w[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR weekly trend turns down
            if close[i] < cam_s1_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR weekly trend turns up
            if close[i] > cam_r1_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_v2"
timeframe = "1d"
leverage = 1.0