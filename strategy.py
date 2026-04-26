#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when price breaks above Camarilla R3 (1d) AND 1d EMA34 uptrend AND volume > 1.5 * volume_ma(20)
- Short when price breaks below Camarilla S3 (1d) AND 1d EMA34 downtrend AND volume > 1.5 * volume_ma(20)
- Uses Camarilla pivot levels from prior 1d (completed bar only) for structure-based breakouts
- 1d EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume confirmation filter ensures institutional participation
- Designed for low frequency (target 12-37 trades/year) to minimize fee drag
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Novelty: Combines Camarilla pivot breakouts with HTF trend and volume filter for BTC/ETH edge in both bull/bear markets
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
    
    # Calculate Camarilla levels from prior 1d bar (completed bar only)
    # Camarilla R3 = close + 1.1 * (high - low) / 2
    # Camarilla S3 = close - 1.1 * (high - low) / 2
    # Note: Using typical Camarilla formulas based on prior day's range
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + 1.1 * (h_1d - l_1d) / 2
    camarilla_s3 = c_1d - 1.1 * (h_1d - l_1d) / 2
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed for structure)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate daily EMA34 for trend filter (needs completed daily candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                        np.where(close > ema_34_1d_aligned, 1, -1), 
                        0)
    
    # Calculate volume confirmation: volume > 1.5 * volume_ma(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_confirm[i])):
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
            # Long: Price breaks above Camarilla R3 AND daily uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and trend_1d[i] == 1 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND daily downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and trend_1d[i] == -1 and volume_confirm[i]:
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

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0