#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 level AND 1d EMA34 uptrend AND volume > 1.5 * volume MA(20)
- Short when price breaks below Camarilla S3 level AND 1d EMA34 downtrend AND volume > 1.5 * volume MA(20)
- Camarilla levels from prior 1d provide precise intraday support/resistance
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite Camarilla level (R3/S3) touch or trend reversal
- Novelty: Combines Camarilla breakouts with HTF trend and volume confirmation for BTC/ETH edge
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
    
    # Load 1d data ONCE before loop for Camarilla levels (structure)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar (completed bar only)
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of prior period
    lookback = 1
    prior_close = pd.Series(df_1d['close'].values).shift(lookback).values
    prior_high = pd.Series(df_1d['high'].values).shift(lookback).values
    prior_low = pd.Series(df_1d['low'].values).shift(lookback).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prior_close + (prior_high - prior_low) * 1.1 / 2
    camarilla_s3 = prior_close - (prior_high - prior_low) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (no additional delay needed for structure)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load daily data ONCE before loop for trend filter (HTF)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume spike filter: volume > 1.5 * volume MA(20)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 34 for EMA, 1 for Camarilla shift)
    start_idx = max(20, 34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
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
            # Long: Price breaks above Camarilla R3 AND daily uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND daily downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR daily trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR daily trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0