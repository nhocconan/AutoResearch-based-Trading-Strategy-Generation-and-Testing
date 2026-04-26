#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) combined with 1d EMA34 trend filter and volume confirmation.
Long when: Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND 1d EMA34 uptrend AND volume > 1.5 * 20-period avg.
Short when: Bull Power < 0 AND Bear Power > 0 (bearish momentum) AND 1d EMA34 downtrend AND volume > 1.5 * 20-period avg.
Exit when: Elder Ray signals reverse (Bull Power <= 0 for long, Bear Power >= 0 for short) OR price crosses EMA13.
Uses discrete 0.25 position size. Targets 12-37 trades/year on 6h timeframe.
Works in bull markets via trend-following momentum and in bear markets via short signals with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 13 for EMA13, 34 for 1d EMA, 20 for volume avg
    start_idx = max(13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        size = 0.25  # Fixed position size
        
        if position == 0:
            # Flat - look for entry with Elder Ray alignment, trend, and volume
            # Long: Bull Power > 0 (strong highs) AND Bear Power < 0 (weak lows) AND 1d EMA34 up AND volume spike
            long_entry = (bull_power[i] > 0) and (bear_power[i] < 0) and \
                       (ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]) and \
                       volume_spike[i]
            # Short: Bull Power < 0 (weak highs) AND Bear Power > 0 (strong lows) AND 1d EMA34 down AND volume spike
            short_entry = (bull_power[i] < 0) and (bear_power[i] > 0) and \
                        (ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]) and \
                        volume_spike[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit when Elder Ray turns bearish OR price crosses below EMA13
            if (bull_power[i] <= 0) or (close[i] < ema13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when Elder Ray turns bullish OR price crosses above EMA13
            if (bear_power[i] >= 0) or (close[i] > ema13[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0