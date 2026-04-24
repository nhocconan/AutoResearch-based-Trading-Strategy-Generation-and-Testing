#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend filter to ensure alignment with weekly trend.
- Donchian channel: 20-period high/low breakouts capture momentum in both bull and bear markets.
- Entry: Long when price breaks above 20-period Donchian high AND price > 1w EMA50 AND volume > 2.0 * 20-period average volume.
         Short when price breaks below 20-period Donchian low AND price < 1w EMA50 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Donchian breakout (long exits on lower band break, short exits on upper band break).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian breakouts work in trending markets (capture trends) and ranging markets (fade false breakouts via volume filter).
- 1w EMA50 provides strong trend filter to avoid counter-trend trades during major moves like 2022 crash.
- Volume spike confirmation ensures breakouts have participation, reducing false signals in low-volume environments.
- Estimated trades: ~60 total over 4 years (~15/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for 20-period Donchian + 50 EMA
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w trend filter: EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1w volume average for confirmation
    if len(df_1w) < 21:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1w = df_1w['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    # Donchian channel (20-period) - calculated on LTF (1d)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 55)  # Need 20 for Donchian, 55 for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Donchian breakout
        if position != 0:
            # Exit long: price breaks below 20-period Donchian low
            if position == 1:
                if curr_low < lowest_low[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above 20-period Donchian high
            elif position == -1:
                if curr_high > highest_high[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: price breaks above 20-period Donchian high AND price > 1w EMA50 AND volume confirmation
            long_breakout = curr_high > highest_high[i]
            long_trend = curr_close > ema50_1w_aligned[i]
            long_volume = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            # Short: price breaks below 20-period Donchian low AND price < 1w EMA50 AND volume confirmation
            short_breakout = curr_low < lowest_low[i]
            short_trend = curr_close < ema50_1w_aligned[i]
            short_volume = curr_volume > 2.0 * vol_ma_20[min(i, len(vol_ma_20)-1)] if len(vol_ma_20) > 0 else False
            
            if long_breakout and long_trend and long_volume:
                signals[i] = 0.25
                position = 1
            elif short_breakout and short_trend and short_volume:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0