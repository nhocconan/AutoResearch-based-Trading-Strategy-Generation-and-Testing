#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 12h HMA rising AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band AND 12h HMA falling AND volume > 1.5x 20-period average.
# Exit when price crosses Donchian midpoint OR ATR-based stoploss (2x ATR).
# Uses discrete position size 0.25. Designed to capture breakouts with trend alignment in both bull and bear markets.
# Target: 75-200 trades over 4 years (19-50/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # === 4h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 4h Indicators: ATR (14) for stoploss ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # === 12h HTF: HMA(21) trend ===
    df_12h = get_htf_data(prices, '12h')
    hma_12h = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        hma_val = hma_12h_aligned[i]
        hma_prev = hma_12h_aligned[i-1] if i > 0 else hma_val
        hma_rising = hma_val > hma_prev
        hma_falling = hma_val < hma_prev
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint OR stoploss hit
            if price < donchian_mid[i] or price <= entry_price - 2.0 * atr[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint OR stoploss hit
            if price > donchian_mid[i] or price >= entry_price + 2.0 * atr[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper band AND HMA rising AND volume spike
            if price > highest_20[i] and hma_rising and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below lower band AND HMA falling AND volume spike
            elif price < lowest_20[i] and hma_falling and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMAs
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    # Handle edge cases for array alignment
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad with NaN to match original length
    result = np.full_like(close, np.nan)
    start_idx = period - half_period + sqrt_period - 1
    end_idx = start_idx + len(hma)
    if end_idx <= len(close):
        result[start_idx:end_idx] = hma
    
    return result

name = "4h_Donchian20_12hHMA21_VolumeSpike_V1"
timeframe = "4h"
leverage = 1.0