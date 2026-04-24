#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for EMA trend and volume spike.
- EMA34 > rising indicates bullish bias, EMA34 < falling indicates bearish bias.
- Entry: Long when price breaks above H3 AND EMA34 rising AND volume > 1.5 * 20-period MA.
         Short when price breaks below L3 AND EMA34 falling AND volume > 1.5 * 20-period MA.
- Exit: Opposite Camarilla breakout (L3 for long, H3 for short) or volume drops below average.
- Uses discrete signal size 0.25 to limit drawdown and reduce fee churn.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_rising = ema34 > np.roll(ema34, 1)  # Today > yesterday
    ema34_falling = ema34 < np.roll(ema34, 1)  # Today < yesterday
    
    # Align 1d EMA34 trend to 4h
    ema34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema34_rising)
    ema34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema34_falling)
    
    # Calculate 1d volume MA (20-period) for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * volume_ma_1d)
    
    # Align 1d volume spike to 4h
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Camarilla levels (H3, L3) from previous 1d bar
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 40  # Need enough 1d bars for EMA34 and shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_rising_aligned[i]) or np.isnan(ema34_falling_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike_aligned[i]:
                # Bullish breakout: price breaks above H3 with rising EMA34
                if curr_high > camarilla_h3_aligned[i] and ema34_rising_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below L3 with falling EMA34
                elif curr_low < camarilla_l3_aligned[i] and ema34_falling_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR volume drops below average
            if curr_low < camarilla_l3_aligned[i] or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR volume drops below average
            if curr_high > camarilla_h3_aligned[i] or not volume_spike_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_1dEMA34Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0