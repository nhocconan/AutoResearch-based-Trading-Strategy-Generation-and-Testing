#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volatility regime filter and volume confirmation.
# Enter long when price breaks above Donchian(20) upper band, 1d ATR ratio < 0.8 (low volatility regime), and volume > 1.5x 20-bar average.
# Enter short when price breaks below Donchian(20) lower band under same conditions.
# Exit when price crosses Donchian midpoint or ATR ratio > 1.2 (volatility expansion).
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid fee drag.
# Low volatility regime filters for choppy markets where breakouts are more reliable; volatility expansion signals trend exhaustion.

name = "4h_DonchianBreakout_1dATRRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and ATR ratio (current ATR / 20-period average ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period average ATR for regime calculation
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # ATR ratio: current ATR relative to its 20-period average
    atr_ratio = np.where(atr_ma_20 > 0, atr_14 / atr_ma_20, 1.0)
    
    # Align 1d ATR ratio to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Volatility regime: ATR ratio < 0.8 = low volatility (favorable for breakouts)
        low_vol_regime = atr_ratio_aligned[i] < 0.8
        # Volatility expansion: ATR ratio > 1.2 = exit condition
        vol_expansion = atr_ratio_aligned[i] > 1.2
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]  # Break below previous period's low
        
        # Exit conditions
        exit_long = close[i] < donchian_mid[i] or vol_expansion
        exit_short = close[i] > donchian_mid[i] or vol_expansion
        
        # Handle entries and exits
        if breakout_up and low_vol_regime and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and low_vol_regime and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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