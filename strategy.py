#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high + 1w HMA21 uptrend + volume > 1.5x 20-day avg
# Short when price breaks below 20-day Donchian low + 1w HMA21 downtrend + volume > 1.5x 20-day avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# 1w HMA21 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~10-25 trades/year on 1d timeframe to avoid overtrading.
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
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMA for half period
    wma_half = np.array([np.nan] * len(close_1w))
    for i in range(half_len, len(close_1w)):
        wma_half[i] = wma(close_1w[i-half_len+1:i+1], half_len)
    
    # Calculate WMA for full period
    wma_full = np.array([np.nan] * len(close_1w))
    for i in range(21, len(close_1w)):
        wma_full[i] = wma(close_1w[i-21+1:i+1], 21)
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = np.array([np.nan] * len(close_1w))
    for i in range(21, len(close_1w)):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma[i] = 2 * wma_half[i] - wma_full[i]
    
    # Final HMA: WMA of raw_hma with sqrt(n) period
    hma_21_1w = np.array([np.nan] * len(close_1w))
    for i in range(int(np.sqrt(21)) + 21, len(close_1w)):
        start_idx = i - int(np.sqrt(21)) + 1
        if start_idx >= 0:
            wma_final = wma(raw_hma[start_idx:i+1], int(np.sqrt(21)))
            if not np.isnan(wma_final):
                hma_21_1w[i] = wma_final[-1] if hasattr(wma_final, '__len__') else wma_final
    
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === 1d Indicator: Donchian(20) ===
    # Donchian high = max(high, 20)
    # Donchian low = min(low, 20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (close > Donchian high)
        # 2. 1w HMA21 uptrend (close > HMA21)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           (close[i] > hma_21_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (close < Donchian low)
        # 2. 1w HMA21 downtrend (close < HMA21)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             (close[i] < hma_21_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_1wHMA21_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0