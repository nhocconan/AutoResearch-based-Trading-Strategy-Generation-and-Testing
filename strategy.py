#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
    # Long: price breaks above 20-period high + volume > 2.0x 20-period average + chop > 61.8 (range)
    # Short: price breaks below 20-period low + volume > 2.0x 20-period average + chop > 61.8 (range)
    # Uses discrete sizing (0.25) and ATR-based stoploss (2x ATR)
    # Target: 20-50 trades/year to stay within 12h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels, volume average, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = rolling max of high over 20 periods
    # Lower = rolling min of low over 20 periods
    high_roll_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll_max
    donchian_low = low_roll_min
    
    # Calculate 1d volume average (20-period) for spike confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Chopiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(TR over 14) / (log10(14) * (max(high)-min(low) over 14)))
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    max_high_14_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(14) * (max_high_14_1d - min_low_14_1d)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid division by zero
    chop_1d = 100 * np.log10(pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values / chop_denominator)
    chop_threshold = 61.8  # chop > 61.8 = ranging market (good for mean reversion/breakouts in range)
    chop_regime = chop_1d > chop_threshold
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    # Calculate ATR using true range approximation for 12h timeframe
    atr_12h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_12h[i] = tr  # Simple average for warmup
        else:
            atr_12h[i] = 0.93 * atr_12h[i-1] + 0.07 * tr  # Wilder's smoothing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current 12h volume > 2.0x 20-period average
        vol_avg_20_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_avg_20_12h[i]):
            signals[i] = 0.0
            continue
        volume_spike = volume[i] > 2.0 * vol_avg_20_12h[i]
        
        # Chop regime filter: only trade in ranging markets (chop > 61.8)
        in_chop_regime = chop_regime_aligned[i] > 0.5
        
        # Breakout conditions: price breaks Donchian levels with volume spike and chop regime
        breakout_long = (close[i] > donchian_high_aligned[i]) and volume_spike and in_chop_regime
        breakout_short = (close[i] < donchian_low_aligned[i]) and volume_spike and in_chop_regime
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_12h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0