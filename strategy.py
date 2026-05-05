#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA21 trend filter + volume confirmation
# Long when price breaks above Donchian upper (20) AND close > 12h HMA21 (uptrend) AND volume spike
# Short when price breaks below Donchian lower (20) AND close < 12h HMA21 (downtrend) AND volume spike
# Uses Donchian channels for structure, HMA21 for smooth trend filter (reduces whipsaw),
# volume spike for conviction. Target: 75-200 total trades over 4 years (19-50/year).
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation).
# Timeframe: 4h (proven timeframe for BTC/ETH with good test performance).

name = "4h_Donchian20_HMA21_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for HMA21
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h HMA21
    close_12h = df_12h['close'].values
    # HMA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(data, window):
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights, 'valid') / weights.sum()
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    if half_len > 0 and sqrt_len > 0:
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        wma_diff = 2 * wma_half - wma_full
        # Pad to original length
        wma_diff_padded = np.full(len(close_12h), np.nan)
        wma_diff_padded[half_len-1:len(wma_diff)+half_len-1] = wma_diff
        hma_21 = wma(wma_diff_padded, sqrt_len)
        # Pad hma result
        hma_21_final = np.full(len(close_12h), np.nan)
        start_idx = sqrt_len - 1
        end_idx = start_idx + len(hma_21)
        if end_idx <= len(close_12h):
            hma_21_final[start_idx:end_idx] = hma_21
        hma_21_12h = hma_21_final
    else:
        hma_21_12h = np.full(len(close_12h), np.nan)
    
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Donchian channels (20-period) on 4h
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND uptrend (close > HMA21) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > hma_21_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND downtrend (close < HMA21) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < hma_21_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian high OR closes below HMA21
            if close[i] < donchian_high[i] or close[i] < hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian low OR closes above HMA21
            if close[i] > donchian_low[i] or close[i] > hma_21_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals