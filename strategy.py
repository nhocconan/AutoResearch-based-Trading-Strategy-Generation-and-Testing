#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d Supertrend(10,3) trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Supertrend trend direction and Donchian channel calculation from prior day.
- Trend Filter: 1d Supertrend must align with breakout direction (long: Supertrend bullish, short: bearish).
- Volume Filter: Current 12h volume > 1.8 * 20-period average 12h volume to confirm strong momentum.
- Entry: Long when close > upper Donchian(20) AND Supertrend bullish AND volume spike.
         Short when close < lower Donchian(20) AND Supertrend bearish AND volume spike.
- Exit: Opposite Donchian break (long exits when close < lower Donchian, short exits when close > upper Donchian).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture strong momentum bursts aligned with daily trend while filtering chop/whipsaws.
- Works in bull markets (trend continuation) and bear markets (trend continuation down).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing (RMA equivalent)."""
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr.iloc[0] = high[0] - low[0]  # First TR is just high-low
    atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    return atr.values

def calculate_supertrend(high, low, close, period=10, multiplier=3):
    """Calculate Supertrend indicator."""
    atr = calculate_atr(high, low, close, period)
    
    # Basic upper and lower bands
    basic_ub = (high + low) / 2 + multiplier * atr
    basic_lb = (high + low) / 2 - multiplier * atr
    
    # Initialize final bands
    final_ub = np.full(len(close), np.nan)
    final_lb = np.full(len(close), np.nan)
    supertrend = np.full(len(close), np.nan)
    trend = np.full(len(close), 1)  # Start with bullish trend
    
    for i in range(period, len(close)):
        # Final upper band
        if basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Final lower band
        if basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Supertrend
        if i == period:
            supertrend[i] = final_ub[i]
            trend[i] = -1  # Bearish
        else:
            if supertrend[i-1] == final_ub[i-1]:
                if close[i] <= final_ub[i]:
                    supertrend[i] = final_ub[i]
                else:
                    supertrend[i] = final_lb[i]
                    trend[i] = 1  # Bullish
            else:
                if close[i] >= final_lb[i]:
                    supertrend[i] = final_lb[i]
                    trend[i] = 1  # Bullish
                else:
                    supertrend[i] = final_ub[i]
                    trend[i] = -1  # Bearish
    
    # For periods before calculation, set to close
    supertrend[:period] = close[:period]
    trend[:period] = 1
    
    return supertrend, trend

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Supertrend(10,3) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    supertrend_1d, trend_1d = calculate_supertrend(high_1d, low_1d, close_1d, period=10, multiplier=3)
    
    # Align Supertrend and trend to 12h timeframe (waits for 1d bar close)
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1d Donchian(20) from prior day (to avoid look-ahead)
    prev_high = df_1d['high'].shift(1).values  # Shifted to avoid look-ahead
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Donchian channels for 1d
    donchian_high_1d = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low_1d = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (waits for 1d bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need 30 for Supertrend/DONCHIAN, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(supertrend_1d_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_donchian = donchian_high_aligned[i]
        lower_donchian = donchian_low_aligned[i]
        supertrend_val = supertrend_1d_aligned[i]
        trend_val = trend_1d_aligned[i]
        
        # Volume spike: current volume > 1.8 * 20-period average volume
        volume_spike = curr_volume > 1.8 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_donchian
        broke_below_lower = curr_close < lower_donchian
        
        # Trend alignment conditions
        bullish_trend = trend_val == 1
        bearish_trend = trend_val == -1
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower Donchian
            if position == 1:
                if curr_close < lower_donchian:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper Donchian
            elif position == -1:
                if curr_close > upper_donchian:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper Donchian AND bullish trend AND volume spike
            long_condition = broke_above_upper and bullish_trend and volume_spike
            
            # Short: break below lower Donchian AND bearish trend AND volume spike
            short_condition = broke_below_lower and bearish_trend and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dSupertrend10_3_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0