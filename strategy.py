#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 AND 1w EMA50 uptrend AND volume > 1.5x volume MA20
- Short when price breaks below Camarilla S3 AND 1w EMA50 downtrend AND volume > 1.5x volume MA20
- Camarilla levels derived from prior 1d session (high, low, close) for structure-based entries
- 1w EMA50 filter ensures alignment with weekly trend to avoid counter-trend whipsaws
- Volume spike filter confirms institutional participation and reduces false breakouts
- Designed for low frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite Camarilla level (R3/S3) touch or trend reversal
- Novelty: Combines Camarilla pivot breakouts with weekly trend and volume spike for BTC/ETH edge in all market regimes
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
    
    # Load 12h data ONCE before loop for Camarilla levels (structure)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels from prior 1d bar (completed daily candle)
    # Camarilla R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    lookback = 1
    cam_r3 = (df_12h['close'].shift(1) + 1.1 * (df_12h['high'].shift(1) - df_12h['low'].shift(1)) / 2).values
    cam_s3 = (df_12h['close'].shift(1) - 1.1 * (df_12h['high'].shift(1) - df_12h['low'].shift(1)) / 2).values
    
    # Align Camarilla levels to 12h timeframe (no additional delay needed for structure)
    cam_r3_aligned = align_htf_to_ltf(prices, df_12h, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_12h, cam_s3)
    
    # Load weekly data ONCE before loop for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter (needs completed weekly candle)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1w = np.where(ema_50_1w_aligned > 0, 
                        np.where(close > ema_50_1w_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.5x volume MA20 for participation
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(trend_1w[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with trend and volume confirmation
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND weekly uptrend AND volume spike
            if close[i] > cam_r3_aligned[i] and trend_1w[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND weekly downtrend AND volume spike
            elif close[i] < cam_s3_aligned[i] and trend_1w[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR weekly trend turns down
            if close[i] < cam_s3_aligned[i] or trend_1w[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR weekly trend turns up
            if close[i] > cam_r3_aligned[i] or trend_1w[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0