#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme reversal with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend direction.
- Williams %R: Measures overbought/oversold levels (-20 to -80 range).
- Extreme reversal: Williams %R < -90 for long, > -10 for short (deep oversold/overbought).
- Trend Filter: 1d EMA50 must align with reversal direction (long: close > EMA50, short: close < EMA50).
- Volume Filter: Current 12h volume > 2.0 * 20-period average 12h volume to confirm strong momentum.
- Entry: Long when Williams %R < -90 AND close > 1d EMA50 AND volume spike.
         Short when Williams %R > -10 AND close < 1d EMA50 AND volume spike.
- Exit: Opposite Williams %R extreme (long exits when Williams %R > -10, short exits when Williams %R < -90).
- Signal size: 0.25 discrete to minimize fee drag.
- Designed to capture exhaustion moves in bear market rallies and bull market pullbacks.
- Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
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
    
    # Calculate 1d Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need min 15 for Williams %R
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R calculation: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe (waits for 1d bar close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h volume average for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        williams_r_val = williams_r_aligned[i]
        ema_50_level = ema_50_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 * 20-period average volume
        volume_spike = curr_volume > 2.0 * vol_ma_20[i]
        
        # Williams %R extreme conditions
        williams_r_extreme_oversold = williams_r_val < -90  # Deep oversold
        williams_r_extreme_overbought = williams_r_val > -10  # Deep overbought
        williams_r_exit_oversold = williams_r_val > -10     # Exit long threshold
        williams_r_exit_overbought = williams_r_val < -90   # Exit short threshold
        
        # Trend alignment conditions
        above_ema = curr_close > ema_50_level
        below_ema = curr_close < ema_50_level
        
        # Exit conditions: opposite Williams %R extreme
        if position != 0:
            # Exit long: Williams %R exits oversold territory (> -10)
            if position == 1:
                if williams_r_exit_oversold:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R exits overbought territory (< -90)
            elif position == -1:
                if williams_r_exit_overbought:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filters
        if position == 0:
            # Long: extreme oversold AND above EMA50 AND volume spike
            long_condition = williams_r_extreme_oversold and above_ema and volume_spike
            
            # Short: extreme overbought AND below EMA50 AND volume spike
            short_condition = williams_r_extreme_overbought and below_ema and volume_spike
            
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

name = "12h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0