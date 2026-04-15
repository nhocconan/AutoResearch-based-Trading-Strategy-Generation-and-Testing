#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band + 12h HMA rising + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower band + 12h HMA falling + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to balance return and drawdown control.
# 12h HMA provides smooth trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-30 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Hull Moving Average (HMA) 21 ===
    def calculate_hma(series, period):
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(arr, window):
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        # Handle edge cases
        if len(series) < period:
            return np.full_like(series, np.nan)
        
        wma_half = wma(series, half_period)
        wma_full = wma(series, period)
        
        # 2 * WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Final WMA of sqrt period
        hma = wma(raw_hma, sqrt_period)
        
        # Pad with NaN to match original length
        result = np.full_like(series, np.nan)
        start_idx = period - 1
        end_idx = start_idx + len(hma)
        result[start_idx:end_idx] = hma
        
        return result
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate HMA using typical price for 12h
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    hma_12h = calculate_hma(typical_price_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate HMA slope (rising/falling)
    hma_slope = np.diff(hma_12h_aligned, prepend=hma_12h_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # === 4h Donchian Channel (20-period) ===
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    upper_band = highest_high
    lower_band = lowest_low
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period, 20) + 30  # Donchian(20) + volume(20) + HMA buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper band (close > upper_band)
        # 2. 12h HMA rising (slope > 0)
        # 3. Volume confirmation
        if (close[i] > upper_band[i]) and \
           hma_rising[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower band (close < lower_band)
        # 2. 12h HMA falling (slope < 0)
        # 3. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             hma_falling[i] and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hHMA_Volume_Filter_v2"
timeframe = "4h"
leverage = 1.0