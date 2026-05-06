#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with 1d ATR regime filter and volume confirmation
# Long when price breaks above 1d Donchian upper (20) AND 1d ATR(14) > ATR(50) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 1d Donchian lower (20) AND 1d ATR(14) > ATR(50) AND volume > 1.5 * avg_volume(20)
# Exit when price touches 1d Donchian midpoint or opposite Donchian level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Donchian provides strong structural breakout levels aligned with daily session
# ATR regime filter ensures we only trade in sufficient volatility (avoid low-vol whipsaws)
# Volume confirmation filters weak breakouts
# Works in bull (trend continuation breakouts) and bear (trend continuation breakdowns)

name = "4h_1dDonchian20_ATRRegime_Volume"
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
    
    # Get 1d data ONCE before loop for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    donchian_upper_1d = high_series_1d.rolling(window=20, min_periods=20).max().values
    donchian_lower_1d = low_series_1d.rolling(window=20, min_periods=20).min().values
    donchian_middle_1d = (donchian_upper_1d + donchian_lower_1d) / 2.0
    
    # Calculate 1d ATR(14) and ATR(50) for regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_regime = atr_14_1d > atr_50_1d  # High volatility regime
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle_1d)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian upper with high volatility regime and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                atr_regime_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian lower with high volatility regime and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  atr_regime_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1d Donchian middle or lower (reversal or profit take)
            if close[i] <= donchian_middle_aligned[i] or close[i] <= donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1d Donchian middle or upper (reversal or profit take)
            if close[i] >= donchian_middle_aligned[i] or close[i] >= donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals