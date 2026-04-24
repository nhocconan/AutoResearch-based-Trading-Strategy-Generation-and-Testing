#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for EMA50 trend filter.
- Donchian levels from prior 20 periods of 12h: upper = max(high, 20), lower = min(low, 20)
  Long when price breaks above upper with volume spike, Short when price breaks below lower with volume spike.
- Trend filter: Only trade in direction of 1w EMA50 (long if EMA50 rising, short if falling).
- Volume confirmation: current volume > 2.0x 20-period volume MA to ensure strong participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying upper breakouts in uptrend, in bear via selling lower breakouts in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if i >= 20:  # Need 20 periods for Donchian
            # Calculate Donchian levels from prior 20 periods of 12h
            donchian_upper = np.max(high[i-20:i])
            donchian_lower = np.min(low[i-20:i])
            
            if position == 0:
                # Only trade in direction of 1w EMA50 trend
                if i > 0 and not np.isnan(ema_50_1w_aligned[i-1]):
                    ema50_slope = ema_50_1w_aligned[i] - ema_50_1w_aligned[i-1]
                    if ema50_slope > 0:  # Uptrend
                        # Long when price breaks above upper with volume spike
                        if close[i] > donchian_upper and volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                    elif ema50_slope < 0:  # Downtrend
                        # Short when price breaks below lower with volume spike
                        if close[i] < donchian_lower and volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
            elif position == 1:
                # Long exit: price breaks below lower or opposite signal
                if close[i] < donchian_lower:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above upper or opposite signal
                if close[i] > donchian_upper:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0