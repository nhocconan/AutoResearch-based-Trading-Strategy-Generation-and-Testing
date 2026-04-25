#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend direction and Donchian channel calculation.
- Donchian Channel: Upper/lower bands from prior 20 periods of 1d high/low.
- Trend Filter: 1d EMA34 must align with breakout direction (long: close > EMA34, short: close < EMA34).
- Volume Filter: Current 4h volume > 1.5 * 20-period average 4h volume to confirm momentum.
- Entry: Long when close > Upper Band AND close > 1d EMA34 AND volume confirmation.
         Short when close < Lower Band AND close < 1d EMA34 AND volume confirmation.
- Exit: Opposite Donchian break (long exits when close < Lower Band, short exits when close > Upper Band).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture breakouts aligned with daily trend while filtering false breakouts in chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian(20) channels from prior day data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior 20-day Donchian channels (shifted to avoid look-ahead)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 4h timeframe (waits for 1d bar close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_max_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().shift(1).values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_band = upper_band_aligned[i]
        lower_band = lower_band_aligned[i]
        ema_34_level = ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        broke_above_upper = curr_close > upper_band
        broke_below_lower = curr_close < lower_band
        
        # Trend alignment conditions
        above_ema = curr_close > ema_34_level
        below_ema = curr_close < ema_34_level
        
        # Exit conditions: opposite Donchian break
        if position != 0:
            # Exit long: close breaks below lower band
            if position == 1:
                if curr_close < lower_band:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: close breaks above upper band
            elif position == -1:
                if curr_close > upper_band:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend and volume filters
        if position == 0:
            # Long: break above upper band AND above EMA34 AND volume confirmation
            long_condition = broke_above_upper and above_ema and volume_confirm
            
            # Short: break below lower band AND below EMA34 AND volume confirmation
            short_condition = broke_below_lower and below_ema and volume_confirm
            
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

name = "4h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0