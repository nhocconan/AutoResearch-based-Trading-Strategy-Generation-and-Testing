#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Donchian(20) high + 1d volume > 1.2x 20-period 1d avg + CHOP(14) > 61.8 (range)
    # Short: price breaks below Donchian(20) low + 1d volume > 1.2x 20-period 1d avg + CHOP(14) > 61.8 (range)
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 20-50 trades/year to stay within 4h optimal range (75-200 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume average for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    # CHOP = 100 * LOG10( sum(ATR(1),14) / (MAX(HIGH,14) - MIN(LOW,14)) ) / LOG10(14)
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
    chop_denominator = max_high_14_1d - min_low_14_1d
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_1d = 100 * np.log10(pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values / chop_denominator) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 4h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR for stoploss
    tr_4h = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr_4h[0] = high[0] - low[0]
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i]) or
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.2x 20-period average (use previous day's volume)
        vol_confirmed = volume_1d[i-1] > 1.2 * vol_avg_20_1d_aligned[i] if i > 0 else False
        
        # Regime filter: CHOP > 61.8 indicates ranging market (good for mean reversion at extremes)
        ranging_regime = chop_1d_aligned[i] > 61.8
        
        # Breakout conditions: price breaks Donchian levels with volume confirmation and ranging regime
        breakout_long = (close[i] > donchian_high_20[i]) and vol_confirmed and ranging_regime
        breakout_short = (close[i] < donchian_low_20[i]) and vol_confirmed and ranging_regime
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_4h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_4h[i]
        
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

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0