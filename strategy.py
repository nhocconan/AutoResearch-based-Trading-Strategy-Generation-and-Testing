#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 level AND 12h EMA50 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Camarilla S3 level AND 12h EMA50 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Camarilla pivot levels from completed 4h bar for structure-based breakouts
- 12h EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike filter confirms institutional participation and reduces false breakouts
- Designed for moderate frequency (target 20-50 trades/year) to minimize fee drag
- Exit on opposite Camarilla level (R3/S3) touch or trend reversal
- Novelty: Combines Camarilla breakouts with HTF trend and volume confirmation for BTC/ETH edge in both bull/bear markets
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
    
    # Load 4h data ONCE before loop for Camarilla levels (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels from prior 4h bar (completed bar only)
    # Camarilla levels based on prior bar's high, low, close
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    # Camarilla R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = c_4h + (h_4h - l_4h) * 1.1 / 4
    camarilla_s3 = c_4h - (h_4h - l_4h) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed for structure)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Load 12h data ONCE before loop for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA50 for trend filter (needs completed 12h candle)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_12h = np.where(ema_50_12h_aligned > 0, 
                         np.where(close > ema_50_12h_aligned, 1, -1), 
                         0)
    
    # Volume spike filter: volume > 1.5 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_12h[i]) or np.isnan(volume_spike[i])):
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
            # Long: Price breaks above Camarilla R3 AND 12h uptrend AND volume spike
            if close[i] > camarilla_r3_aligned[i] and trend_12h[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 12h downtrend AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and trend_12h[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 12h trend turns down
            if close[i] < camarilla_s3_aligned[i] or trend_12h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 12h trend turns up
            if close[i] > camarilla_r3_aligned[i] or trend_12h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0