#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA50 trend direction, 1d for volume confirmation.
- EMA50 > rising (current > previous) indicates bullish bias, EMA50 < falling indicates bearish bias.
- Entry: Long when price breaks above Donchian upper (20-period high) AND EMA50 bullish AND volume spike.
         Short when price breaks below Donchian lower (20-period low) AND EMA50 bearish AND volume spike.
- Exit: Opposite Donchian breakout (price crosses midline) or EMA50 trend reversal.
- Volume confirmation: current volume > 2.0 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA50 trend: bullish if current > previous, bearish if current < previous
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = ema_50[0]  # first value
    ema_bullish = ema_50 > ema_50_prev
    ema_bearish = ema_50 < ema_50_prev
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    volume_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    # Align volume MA to 6h
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Calculate Donchian channels (20-period) on 6h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # Align HTF indicators to 6h
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish.astype(float))
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA50 and 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_bullish_aligned[i]) or np.isnan(ema_bearish_aligned[i]) or 
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        vol_ma = volume_ma_20_aligned[i]
        
        # Volume confirmation: current volume > 2.0 * 20-period volume MA
        volume_spike = curr_volume > (2.0 * vol_ma)
        
        upper = high_roll[i]
        lower = low_roll[i]
        mid = donchian_mid[i]
        
        ema_bull = ema_bullish_aligned[i] > 0.5
        ema_bear = ema_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Check for entry signals
            if volume_spike:
                # Bullish breakout: price breaks above upper Donchian AND EMA50 bullish
                if curr_high > upper and ema_bull:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian AND EMA50 bearish
                elif curr_low < lower and ema_bear:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price crosses below midline OR EMA50 turns bearish
            if curr_close < mid or ema_bear:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above midline OR EMA50 turns bullish
            if curr_close > mid or ema_bull:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50Trend_1dVolumeSpike_v1"
timeframe = "6h"
leverage = 1.0