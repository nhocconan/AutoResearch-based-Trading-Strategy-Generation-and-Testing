#!/usr/bin/env python3
"""
4h_KAMA_Trend_Donchian20_Exit_VolumeRegime
Hypothesis: 4h KAMA trend direction + Donchian(20) breakout entry with volume regime filter.
Long when KAMA trending up and price breaks above Donchian upper band with moderate volume confirmation.
Short when KAMA trending down and price breaks below Donchian lower band with moderate volume confirmation.
Exit when price crosses KAMA (trend reversal) or Donchian opposite band.
Uses volume regime: avoids low-volume chop, requires volume > 1.2x 50-bar average but < 4x to avoid spikes.
Designed for 20-40 trades/year on 4h with clear trend-following logic and controlled frequency.
Works in bull markets via trend continuation and in bear markets via defined trend exits.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA calculation (ER=10, fast=2, slow=30)
    def kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.full_like(close, np.nan, dtype=float)
        kama[er_period] = close[er_period]
        for i in range(er_period + 1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # Get daily data for trend confirmation (optional HTF filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA trend (primary timeframe)
    kama_vals = kama(close, er_period=10, fast=2, slow=30)
    kama_trend_up = kama_vals > np.roll(kama_vals, 1)
    kama_trend_down = kama_vals < np.roll(kama_vals, 1)
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume regime filter: avoid chop, require moderate volume
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_regime = (volume > 1.2 * vol_ma_50) & (volume < 4.0 * vol_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_vals[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Regime-based entry logic
            if ema_50_1d_aligned[i] > 0:  # Daily trend filter (always true after warmup)
                # Long: KAMA trending up + break above Donchian high + volume regime
                long_signal = kama_trend_up[i] and (close[i] > donchian_high[i]) and vol_regime[i]
                # Short: KAMA trending down + break below Donchian low + volume regime
                short_signal = kama_trend_down[i] and (close[i] < donchian_low[i]) and vol_regime[i]
            else:
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA trend reversal OR price crosses Donchian low
            exit_signal = (not kama_trend_up[i]) or (close[i] < donchian_low[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA trend reversal OR price crosses Donchian high
            exit_signal = (not kama_trend_down[i]) or (close[i] > donchian_high[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Trend_Donchian20_Exit_VolumeRegime"
timeframe = "4h"
leverage = 1.0