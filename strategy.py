#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R3 level AND 1d EMA50 uptrend AND volume > 2.0 * volume_ma(20)
- Short when price breaks below Camarilla S3 level AND 1d EMA50 downtrend AND volume > 2.0 * volume_ma(20)
- Uses Camarilla pivot levels from 12h chart for structure-based breakouts
- 1d EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws in bear markets
- Volume spike (2.0x) confirms institutional participation and reduces false breakouts
- Exit on opposite Camarilla level (S3 for longs, R3 for shorts) or trend reversal
- Designed for lower frequency (target 12-37 trades/year on 12h) to minimize fee drag
- Novelty: Applying proven Camarilla R3/S3 breakout logic to 12h timeframe with 1d trend filter
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
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate Camarilla pivot levels on 12h chart (primary timeframe)
    # Using previous bar's OHLC for Camarilla calculation
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current values
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla calculations
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Resistance levels
    R3 = pivot + (range_hl * 1.1 / 4.0)
    R4 = pivot + (range_hl * 1.1 / 2.0)
    # Support levels
    S3 = pivot - (range_hl * 1.1 / 4.0)
    S4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Calculate volume filter: volume > 2.0 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA, 20 for volume MA)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(trend_1d[i]) or np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if close[i] > R3[i] and trend_1d[i] == 1 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif close[i] < S3[i] and trend_1d[i] == -1 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3 OR 1d trend turns down
            if close[i] < S3[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3 OR 1d trend turns up
            if close[i] > R3[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0