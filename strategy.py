#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with volume confirmation and 1w EMA50 trend filter.
# Long when price breaks above 1d Donchian high AND volume > 1.5x 20-period average AND close > 1w EMA50 (uptrend).
# Short when price breaks below 1d Donchian low AND volume > 1.5x 20-period average AND close < 1w EMA50 (downtrend).
# Exit when price crosses the opposite Donchian level or volume drops below average.
# Uses discrete position size 0.25. Donchian channels provide clear breakout levels, volume confirms conviction,
# and 1w EMA50 ensures alignment with higher timeframe trend to avoid whipsaws in choppy markets.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (capture uptrend breakouts) and bear markets (capture downtrend breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === 1d Indicators: Donchian(20) channels ===
    # Donchian High = max(high, lookback=20)
    # Donchian Low = min(low, lookback=20)
    lookback = 20
    donchian_high_1d = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low_1d = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1w Indicators: EMA50 for trend filter ===
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 and Donchian need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        ema50 = ema50_aligned[i]
        vol_avg = vol_avg_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian Low (breakdown) OR volume < average (loss of conviction)
            if (price < donchian_low) or (vol < vol_avg):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian High (breakout) OR volume < average (loss of conviction)
            if (price > donchian_high) or (vol < vol_avg):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian High AND Volume > 1.5x average AND Price > EMA50 (uptrend)
            if (price > donchian_high) and (vol > 1.5 * vol_avg) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Donchian Low AND Volume > 1.5x average AND Price < EMA50 (downtrend)
            elif (price < donchian_low) and (vol > 1.5 * vol_avg) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_1dDonchian20_1wEMA50_VolumeConfirmation_V1"
timeframe = "4h"
leverage = 1.0