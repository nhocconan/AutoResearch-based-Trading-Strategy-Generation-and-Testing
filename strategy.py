#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Supertrend for trend direction, 12h Donchian(20) breakout for entry,
# and 12h volume spike (>2.0x 20-bar avg) for confirmation. Long when price breaks above Donchian upper
# band AND 1d Supertrend is bullish AND volume spike. Short when price breaks below Donchian lower band
# AND 1d Supertrend is bearish AND volume spike. Exit on opposite Donchian breakout or volume drop.
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 total trades over 4 years.
# Supertrend filters for primary trend alignment to avoid counter-trend trades.
# Donchian breakout provides clear entry/exit levels with built-in volatility adaptation.
# Volume spike confirms institutional participation and reduces false breakouts.
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend).

name = "12h_Supertrend_Donchian20_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Supertrend calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:  # Need enough for ATR and Supertrend
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend(10, 3.0) on daily timeframe
    # ATR
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1d + low_1d) / 2 + 3.0 * atr
    basic_lb = (high_1d + low_1d) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(close_1d))
    final_lb = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        if basic_ub[i] < final_ub[i-1] or close_1d[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        if basic_lb[i] > final_lb[i-1] or close_1d[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb[i]
        else:
            final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        if supertrend[i-1] == final_ub[i-1]:
            supertrend[i] = final_ub[i] if close_1d[i] <= final_ub[i] else final_lb[i]
        else:
            supertrend[i] = final_lb[i] if close_1d[i] >= final_lb[i] else final_ub[i]
    
    # Supertrend trend direction: 1 for bullish (price > supertrend), -1 for bearish (price < supertrend)
    supertrend_dir = np.where(close_1d > supertrend, 1, -1)
    
    # Align Supertrend direction to 12h timeframe (wait for completed daily bar)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, supertrend_dir)
    
    # Get 12h data ONCE before loop for Donchian(20) calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough for Donchian
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (no additional delay needed)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band, 1d Supertrend bullish, volume confirmation, in session
            if (close[i] > donchian_high_aligned[i] and supertrend_dir_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band, 1d Supertrend bearish, volume confirmation, in session
            elif (close[i] < donchian_low_aligned[i] and supertrend_dir_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian lower band OR volume drops below average
            if close[i] < donchian_low_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian upper band OR volume drops below average
            if close[i] > donchian_high_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals