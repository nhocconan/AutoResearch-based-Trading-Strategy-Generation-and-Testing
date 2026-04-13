#!/usr/bin/env python3
"""
Hypothesis: 1-day 1-week Williams Fractal reversal with 1-week volatility regime and 1-day volume confirmation.
Uses 1-week bearish/bullish fractals for reversal signals, 1-week ATR ratio < 0.8 (low volatility) to filter false signals,
and 1-day volume > 1.5x 20-period average to confirm conviction. Long on bullish fractal break of recent high in low vol with volume.
Short on bearish fractal break of recent low in low vol with volume. Target: 30-100 total trades over 4 years (7-25/year).
Williams Fractals work well in ranging/mean-reverting markets (2025-2026 test) and capture reversals in trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume spike (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 1w data for Williams Fractals and volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Williams Fractals (5-bar: center bar with 2 lower highs/lows on each side)
    # Bearish fractal: high[n-2] < high[n] and high[n-1] < high[n] and high[n+1] < high[n] and high[n+2] < high[n]
    # Bullish fractal: low[n-2] > low[n] and low[n-1] > low[n] and low[n+1] > low[n] and low[n+2] > low[n]
    n_1w = len(high_1w)
    bearish_fractal = np.zeros(n_1w, dtype=bool)
    bullish_fractal = np.zeros(n_1w, dtype=bool)
    
    for i in range(2, n_1w - 2):
        if (high_1w[i-2] < high_1w[i] and high_1w[i-1] < high_1w[i] and 
            high_1w[i+1] < high_1w[i] and high_1w[i+2] < high_1w[i]):
            bearish_fractal[i] = True
        if (low_1w[i-2] > low_1w[i] and low_1w[i-1] > low_1w[i] and 
            low_1w[i+1] > low_1w[i] and low_1w[i+2] > low_1w[i]):
            bullish_fractal[i] = True
    
    # Calculate 1-week ATR for volatility regime
    # TR = max(high-low, |high-close_prev|, |low-close_prev|)
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-week ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1w / atr_ma_50
    
    # Volatility regime: ATR ratio < 0.8 = low volatility (good for reversals)
    low_volatility = atr_ratio < 0.8
    
    # Align HTF data to LTF
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bearish_fractal.astype(float), additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1w, bullish_fractal.astype(float), additional_delay_bars=2)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    low_volatility_aligned = align_htf_to_ltf(prices, df_1w, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or 
            np.isnan(bullish_fractal_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(low_volatility_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Williams Fractal + volume spike + low volatility
        bullish_fractal_signal = bullish_fractal_aligned[i] > 0.5
        bearish_fractal_signal = bearish_fractal_aligned[i] > 0.5
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        vol_regime = low_volatility_aligned[i] > 0.5  # True if low volatility
        
        # Additional price confirmation: price must break recent swing point
        # For bullish: price > recent swing high (using 5-period lookback high)
        # For bearish: price < recent swing low (using 5-period lookback low)
        lookback = 5
        if i >= lookback:
            recent_high = np.max(high[i-lookback:i+1])
            recent_low = np.min(low[i-lookback:i+1])
            price_conf_long = close[i] > recent_high
            price_conf_short = close[i] < recent_low
        else:
            price_conf_long = False
            price_conf_short = False
        
        long_entry = bullish_fractal_signal and vol_confirm and vol_regime and price_conf_long
        short_entry = bearish_fractal_signal and vol_confirm and vol_regime and price_conf_short
        
        # Exit on opposite fractal signal (mean reversion)
        exit_long = position == 1 and bearish_fractal_signal
        exit_short = position == -1 and bullish_fractal_signal
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williams_fractal_vol_vol"
timeframe = "1d"
leverage = 1.0