#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + Donchian breakout
# Uses Choppiness Index (14) to detect trending (CHOP < 38.2) vs ranging (CHOP > 61.8) markets
# In trending markets: break above Donchian upper (20) = long, break below Donchian lower (20) = short
# In ranging markets: mean reversion at Donchian channels (touch upper = short, touch lower = long)
# Volume confirmation: volume > 1.5x 24-period average (2 days of 12h bars)
# Designed for 12h timeframe to work in both bull (trend follow) and bear (mean revert) markets
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for higher timeframe context (trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate Donchian channels (20-period) on 1d timeframe for structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Calculate Choppiness Index (14) on 1d timeframe for regime detection
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], 
                     np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index 0
    
    atr1 = tr1
    sum_atr1 = np.nancumsum(atr1)  # cumulative sum ignoring NaN
    
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    chop = np.full(len(high_1d), np.nan)
    for i in range(14, len(high_1d)):
        if not np.isnan(sum_atr1[i]) and atr14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr1[i] / (14 * atr14[i])) / np.log10(14)
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 24-period average volume for spike detection (2 days of 12h bars)
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(vol_period, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine regime from Choppiness Index
        ranging = chop_aligned[i] > 61.8
        trending = chop_aligned[i] < 38.2
        
        # Volume confirmation
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            if ranging and volume_confirmation:
                # In ranging market: mean reversion at Donchian channels
                if price <= donch_low_aligned[i] * 1.001:  # touch lower band -> long
                    signals[i] = size
                    position = 1
                elif price >= donch_high_aligned[i] * 0.999:  # touch upper band -> short
                    signals[i] = -size
                    position = -1
            elif trending and volume_confirmation:
                # In trending market: breakout direction
                if price > donch_high_aligned[i]:  # break above upper -> long
                    signals[i] = size
                    position = 1
                elif price < donch_low_aligned[i]:  # break below lower -> short
                    signals[i] = -size
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: opposite Donchian touch or trend change
            if ranging:
                if price >= donch_high_aligned[i] * 0.999:  # touch upper band -> exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:  # trending
                if price < donch_low_aligned[i]:  # break below lower -> exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short exit: opposite Donchian touch or trend change
            if ranging:
                if price <= donch_low_aligned[i] * 1.001:  # touch lower band -> exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:  # trending
                if price > donch_high_aligned[i]:  # break above upper -> exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "12h_ChopRegime_Donchian20_Volume"
timeframe = "12h"
leverage = 1.0