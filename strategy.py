#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel breakout with 1w EMA200 trend filter and volume confirmation.
# Enter long when price breaks above 1d Donchian upper (20) with volume > 2.0x 20-bar average and close > 1w EMA200.
# Enter short when price breaks below 1d Donchian lower (20) with volume > 2.0x average and close < 1w EMA200.
# Exit when price returns to the 1d Donchian midpoint.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn.
# Target: 80-160 total trades over 4 years (20-40/year) to avoid fee drag.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# Uses 1d Donchian for structure (proven edge) and 1w EMA200 for strong trend filter (reduces whipsaws in chop).

name = "4h_Donchian_1dBreakout_1wEMA200_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation (MTF structure)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian upper and lower (20-period)
    donch_hi = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_hi + donch_lo) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donch_hi_aligned = align_htf_to_ltf(prices, df_1d, donch_hi)
    donch_lo_aligned = align_htf_to_ltf(prices, df_1d, donch_lo)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Get 1w data for EMA200 trend filter (MTF trend)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_hi_aligned[i]) or np.isnan(donch_lo_aligned[i]) or np.isnan(donch_mid_aligned[i]) or
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1w EMA200 bias
        bullish_bias = close[i] > ema_200_1w_aligned[i]
        bearish_bias = close[i] < ema_200_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donch_hi_aligned[i]
        short_breakout = close[i] < donch_lo_aligned[i]
        
        # Exit condition: return to midpoint
        long_exit = close[i] < donch_mid_aligned[i]
        short_exit = close[i] > donch_mid_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals