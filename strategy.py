#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with volume confirmation and chop regime filter.
# Weekly Donchian(20) provides strong structural support/resistance based on weekly price action.
# Breakout above weekly Donchian high (long) or below weekly Donchian low (short) with volume spike (>2.0x 20-bar average) for confirmation.
# Chop regime filter: only trade when market is trending (CHOP < 38.2) to avoid false breakouts in ranging markets.
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.
# Target: 30-100 total trades over 4 years = 7-25/year for 1d (within proven winning range).

name = "1d_Donchian20_1w_VolumeSpike_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Donchian high: max(high_1w, window=20)
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Weekly Donchian low: min(low_1w, window=20)
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Get daily data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate chop regime filter (CHOP < 38.2 = trending market)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop calculation: CHOP = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_sum_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    range_14 = highest_high_14 - lowest_low_14
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    chop = 100 * np.log10(atr_sum_14 / range_14) / np.log10(14)
    
    # Chop regime: trending when CHOP < 38.2
    chop_trending = chop < 38.2
    
    # Calculate daily volume spike: >2.0x 20-bar average volume (stricter to reduce trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_trending[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions with volume confirmation and chop regime filter
        long_breakout = close[i] > donchian_high_aligned[i] and volume_spike[i] and chop_trending[i]
        short_breakout = close[i] < donchian_low_aligned[i] and volume_spike[i] and chop_trending[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
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