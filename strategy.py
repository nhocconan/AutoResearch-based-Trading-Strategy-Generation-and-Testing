#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation
# Long when price breaks above Donchian upper (20-day high) + 1w HMA21 uptrend + volume > 1.5x 20-day avg
# Short when price breaks below Donchian lower (20-day low) + 1w HMA21 downtrend + volume > 1.5x 20-day avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1w HMA21 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-30 trades/year on 1d timeframe to avoid overtrading.
# Donchian channels provide clear structure-based entries that work in ranging and trending markets.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # === 1w Indicator: HMA21 ===
    close_1w = df_1w['close'].values
    # Calculate HMA: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    wma_half = np.array([np.nan] * len(close_1w))
    wma_full = np.array([np.nan] * len(close_1w))
    
    for i in range(half_len, len(close_1w)):
        wma_half[i] = wma(close_1w[i-half_len+1:i+1], half_len)
    
    for i in range(21, len(close_1w)):
        wma_full[i] = wma(close_1w[i-21+1:i+1], 21)
    
    raw_hma = 2 * wma_half - wma_full
    hma_21_1w = np.array([np.nan] * len(close_1w))
    
    for i in range(sqrt_len, len(raw_hma)):
        if not np.isnan(raw_hma[i]):
            hma_21_1w[i] = wma(raw_hma[i-sqrt_len+1:i+1], sqrt_len)
    
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Donchian Channel (20-period) ===
    # Upper = highest high of last 20 periods
    # Lower = lowest low of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > highest_high)
        # 2. 1w HMA21 uptrend (close > HMA21)
        # 3. Volume confirmation
        if (close[i] > highest_high[i]) and \
           (close[i] > hma_21_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lowest_low)
        # 2. 1w HMA21 downtrend (close < HMA21)
        # 3. Volume confirmation
        elif (close[i] < lowest_low[i]) and \
             (close[i] < hma_21_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wHMA21_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0