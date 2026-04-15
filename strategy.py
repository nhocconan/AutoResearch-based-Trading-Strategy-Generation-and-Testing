#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20) + 12h HMA21 uptrend + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower (20) + 12h HMA21 downtrend + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 12h HMA21 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 4h timeframe.
# Donchian channels provide clear breakout levels that work in both trending and ranging markets.

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
    
    # === 12h Indicator: HMA21 ===
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        if len(values) < window:
            return np.full_like(values, np.nan)
        wma_half = wma(values, half_window)
        wma_full = wma(values, window)
        raw_hma = 2 * wma_half - wma_full
        # Pad the beginning with NaN to match original length
        padded_raw = np.full(len(values), np.nan)
        padded_raw[half_window-1:len(raw_hma)+half_window-1] = raw_hma
        return wma(padded_raw, sqrt_window)
    
    close_12h = df_12h['close'].values
    hma_21_12h = hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # === 4h Donchian Channels (20) ===
    def donchian_channels(high, low, window):
        upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # HMA21(12h needs 50 bars) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. 12h HMA21 uptrend (close > HMA21)
        # 3. Volume confirmation
        if (close[i] > upper_20[i]) and \
           (close[i] > hma_21_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. 12h HMA21 downtrend (close < HMA21)
        # 3. Volume confirmation
        elif (close[i] < lower_20[i]) and \
             (close[i] < hma_21_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hHMA21_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0