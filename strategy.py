#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Choppiness Index to filter market regime and 4h Donchian breakout for entry
# - Uses 1d Choppiness Index to identify trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets
# - In trending markets (CHOP < 38.2), enters long on breakout above 20-period Donchian high
# - In trending markets (CHOP < 38.2), enters short on breakout below 20-period Donchian low
# - Uses volume spike (volume > 2x 20-period average) for entry confirmation
# - Exits when price crosses back below/above the opposite Donchian band
# - Designed to work in both bull and bear markets by only trading in clear trends
# - Target: 20-50 total trades over 4 years (5-12/year) with 0.30 position sizing

name = "4h_Choppiness_Donchian_Breakout"
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
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d True Range (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index (14-period)
    # CHOP = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr14 / (hh14 - ll14 + 1e-10)) / np.log10(14)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Choppiness Index to 4h timeframe
    chop_4h = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align 4h Donchian channels to 4h timeframe (already aligned, but using helper for consistency)
    donch_high_4h = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_4h = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)  # Strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(chop_4h[i]) or np.isnan(donch_high_4h[i]) or 
            np.isnan(donch_low_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for trending market (Choppiness Index < 38.2)
            trending_market = chop_4h[i] < 38.2
            
            if trending_market:
                # Long: price breaks above Donchian high with volume spike
                if close[i] > donch_high_4h[i] and volume_spike[i]:
                    signals[i] = 0.30
                    position = 1
                # Short: price breaks below Donchian low with volume spike
                elif close[i] < donch_low_4h[i] and volume_spike[i]:
                    signals[i] = -0.30
                    position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donch_low_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donch_high_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals