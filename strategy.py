#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA50 trend direction.
- EMA50 > rising: bullish trend bias, EMA50 < falling: bearish trend bias.
- Entry: Long when price breaks above Donchian upper (20) AND EMA50 trending up AND volume spike.
         Short when price breaks below Donchian lower (20) AND EMA50 trending down AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Volume confirmation: current volume > 2.0 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA50 slope: rising if current > previous, falling if current < previous
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    
    # Align HTF indicators to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), donchian_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_50_slope)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 12h bars for EMA50 and 20 for Donchian/volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        ema_50_slope_val = ema_50_slope_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price breaks above upper AND EMA50 trending up
                if curr_high > upper and ema_50_slope_val > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower AND EMA50 trending down
                elif curr_low < lower and ema_50_slope_val < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below lower OR loss of volume confirmation OR EMA50 turns down
            if curr_low < lower or not volume_spike[i] or ema_50_slope_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper OR loss of volume confirmation OR EMA50 turns up
            if curr_high > upper or not volume_spike[i] or ema_50_slope_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0