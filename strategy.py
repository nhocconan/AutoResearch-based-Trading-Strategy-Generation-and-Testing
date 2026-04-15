#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) + 12h HMA rising (bullish) + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower (20-period) + 12h HMA falling (bearish) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# HMA provides smooth trend detection with less lag than SMA/EMA. Works in trending markets (bull/bear) while avoiding chop.
# Volume confirmation ensures breakouts have participation, reducing false signals.

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
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicator: Hull Moving Average (HMA) for trend direction ===
    # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        
        wma_half = wma(values, half_window)
        wma_full = wma(values, window)
        
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(values, np.nan)
        
        # 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
        
        # WMA(sqrt(n)) of the above
        hma_values = wma(raw_hma, sqrt_window)
        
        # Pad with NaN to match original length
        result = np.full_like(values, np.nan)
        start_idx = window - half_window - sqrt_window + 1
        end_idx = start_idx + len(hma_values)
        if start_idx >= 0 and end_idx <= len(values):
            result[start_idx:end_idx] = hma_values
        return result
    
    close_12h = df_12h['close'].values
    hma_12h = hma(close_12h, 21)  # 21-period HMA
    
    # Calculate HMA slope (rising/falling)
    hma_slope = np.diff(hma_12h, prepend=np.nan)
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    hma_rising_aligned = align_htf_to_ltf(prices, df_12h, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_12h, hma_falling)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 21) + 20  # Donchian(20) + HMA(21) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Trend (12h HMA rising)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           hma_rising_aligned[i] and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Trend (12h HMA falling)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             hma_falling_aligned[i] and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hHMA_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0