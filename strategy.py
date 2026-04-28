#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and ATR-based trend filter.
# Enter long when price breaks above 1d Donchian upper channel with volume > 2.0x average and ATR(14) rising.
# Enter short when price breaks below 1d Donchian lower channel with volume > 2.0x average and ATR(14) falling.
# Exit when price returns to the 1d Donchian midpoint or touches the opposite channel.
# Uses discrete position sizing (0.25) to control risk and minimize fee churn. Target: 50-120 total trades over 4 years.
# Works in bull markets (breakouts continue up with rising volatility) and bear markets (breakdowns continue down with falling volatility).
# Uses 1d Donchian for structure (more stable than lower timeframes) and ATR(14) slope for trend filter (adaptive to volatility regimes).

name = "4h_Donchian20_1d_Volume_ATRTrend_v1"
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
    
    # Donchian upper and lower channels (based on previous 20 bars)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate 1d ATR(14) for trend filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR slope (rising/falling trend filter)
    atr_slope = atr_14 - np.roll(atr_14, 5)  # 5-bar slope
    atr_slope[0:5] = 0  # first 5 bars
    
    # Align ATR and slope to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_slope_aligned = align_htf_to_ltf(prices, df_1d, atr_slope)
    
    # Calculate volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_slope_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: ATR slope (rising for long, falling for short)
        bullish_bias = atr_slope_aligned[i] > 0
        bearish_bias = atr_slope_aligned[i] < 0
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # Exit conditions: return to midpoint or touch opposite channel
        long_exit = close[i] < donchian_mid_aligned[i]
        short_exit = close[i] > donchian_mid_aligned[i]
        
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