#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index Regime Filter + Donchian(20) Breakout + Volume Confirmation
# In trending markets (CHOP < 38.2): trade Donchian breakouts (long at 20-period high, short at low)
# In ranging markets (CHOP > 61.8): fade reversals at Bollinger Bands (2,2) - short at upper band, long at lower
# Volume confirmation requires current volume > 1.5x 20-period median volume
# Works in bull markets (trend following breakouts) and bear markets (trend following breakdowns or mean reversion in ranges)
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (14-period) on daily
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Sum of true range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    chop = np.where((hh - ll) == 0, 50, chop)  # Avoid division by zero
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Load 1d data for Bollinger Bands (for ranging market signals)
    bb_period = 20
    bb_std = 2.0
    sma = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + (bb_std * std)
    lower_bb = sma - (bb_std * std)
    
    # Align Bollinger Bands to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate Donchian Channels (20-period) on 12h
    # Load 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian Channels
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian Channels to 12h timeframe (no additional delay needed)
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period median
        vol_median = np.median(volume[max(0, i-19):i+1]) if i >= 20 else np.median(volume[:i+1])
        vol_confirm = volume[i] > 1.5 * vol_median
        
        # Regime-based trading logic
        if chop_aligned[i] < 38.2:  # Trending market - follow Donchian breakouts
            # Long: price breaks above Donchian high
            if close[i] > donch_high_aligned[i] and vol_confirm and position <= 0:
                position = 1
                signals[i] = base_size
            # Short: price breaks below Donchian low
            elif close[i] < donch_low_aligned[i] and vol_confirm and position >= 0:
                position = -1
                signals[i] = -base_size
            # Exit trending position on opposite Donchian breakout
            elif position == 1 and close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
                
        elif chop_aligned[i] > 61.8:  # Ranging market - fade at Bollinger Bands
            # Short at upper Bollinger Band (fade resistance)
            if close[i] > upper_bb_aligned[i] and vol_confirm and position >= 0:
                position = -1
                signals[i] = -base_size
            # Long at lower Bollinger Band (fade support)
            elif close[i] < lower_bb_aligned[i] and vol_confirm and position <= 0:
                position = 1
                signals[i] = base_size
            # Exit ranging position when price returns to middle (SMA)
            elif position == 1 and close[i] > sma[min(i, len(sma)-1)] if not np.isnan(sma[min(i, len(sma)-1)]) else False:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] < sma[min(i, len(sma)-1)] if not np.isnan(sma[min(i, len(sma)-1)]) else False:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Chop_Donchian_BB_Volume"
timeframe = "12h"
leverage = 1.0