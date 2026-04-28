#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly ATR-based volatility regime filter combined with daily Donchian breakout.
# Long when: price breaks above daily Donchian upper (20) AND weekly ATR ratio (current/50-period) > 1.2 (high volatility regime)
# Short when: price breaks below daily Donchian lower (20) AND weekly ATR ratio > 1.2
# Uses discrete sizing (0.25) to limit fee drag. Target: 30-80 trades/year.
# Weekly ATR filter ensures we only trade during expansion phases, avoiding low-volatility whipsaws.
# Donchian provides clear breakout levels from higher timeframe. Works in bull (breakouts with trend) and bear (failed breaks reverse via exits).

name = "6h_WeeklyATR_VolRegime_DonchianBreakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels (MTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    n_1d = len(high_1d)
    donchian_high = np.full(n_1d, np.nan)
    donchian_low = np.full(n_1d, np.nan)
    
    for i in range(19, n_1d):  # min_periods=20
        donchian_high[i] = np.max(high_1d[i-19:i+1])
        donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    # Forward fill Donchian levels
    donchian_high = pd.Series(donchian_high).ffill().values
    donchian_low = pd.Series(donchian_low).ffill().values
    
    # Align 1d Donchian to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Get weekly data for ATR regime filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly True Range and ATR(50)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    n_1w = len(high_1w)
    tr_1w = np.zeros(n_1w)
    atr_1w = np.zeros(n_1w)
    
    for i in range(1, n_1w):
        tr = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
        tr_1w[i] = tr
    
    # Calculate ATR(50) with min_periods=50
    for i in range(50, n_1w):
        atr_1w[i] = np.mean(tr_1w[i-49:i+1])
    
    # Forward fill ATR
    atr_1w = pd.Series(atr_1w).ffill().values
    
    # Calculate weekly ATR ratio: current ATR / 50-period ATR mean (using prior 50 weeks)
    atr_ma_50 = np.full(n_1w, np.nan)
    for i in range(100, n_1w):  # min_periods=100 for stability
        atr_ma_50[i] = np.mean(atr_1w[i-99:i+1])
    
    atr_ratio = np.full(n_1w, np.nan)
    for i in range(100, n_1w):
        if atr_ma_50[i] > 0:
            atr_ratio[i] = atr_1w[i] / atr_ma_50[i]
    
    # Forward fill ATR ratio
    atr_ratio = pd.Series(atr_ratio).ffill().values
    
    # Align weekly ATR ratio to 6h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1w, atr_ratio)
    
    # Volume confirmation: 6h volume > 1.5x 24-period average (4 days)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # High volatility regime: weekly ATR ratio > 1.2
        high_vol_regime = atr_ratio_aligned[i] > 1.2
        
        # Donchian breakout conditions with volume confirmation and volatility filter
        long_breakout = close[i] > donchian_high_aligned[i] and volume_spike[i] and high_vol_regime
        short_breakout = close[i] < donchian_low_aligned[i] and volume_spike[i] and high_vol_regime
        
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