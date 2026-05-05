#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d HMA21 trend filter + volume confirmation
# Long when price breaks above Donchian upper(20) AND close > 1d HMA21 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower(20) AND close < 1d HMA21 AND volume > 1.5x 20-period average
# Exit when price crosses 1d HMA21 (trend reversal) OR touches opposite Donchian band
# Uses 4h primary timeframe with 1d HTF for trend filter to capture sustained moves across market regimes
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag
# Donchian provides clear breakout levels; 1d HMA21 filters for higher-timeframe trend; volume confirms participation
# Works in bull markets via breakouts and in bear markets via trend-filtered shorts

name = "4h_Donchian20_Breakout_1dHMA21_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for HMA21 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate HMA21 on 1d close for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    close_1d = df_1d['close'].values
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    # Pad arrays to match original length
    wma_half_padded = np.concatenate([np.full(half_len, np.nan), wma_half])
    wma_full_padded = np.concatenate([np.full(21 - 1, np.nan), wma_full])
    raw_hma = 2 * wma_half_padded - wma_full_padded
    hma_21_1d = wma(raw_hma, sqrt_len)
    # Pad HMA result
    hma_21_1d_padded = np.concatenate([np.full(sqrt_len - 1, np.nan), hma_21_1d])
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d_padded)
    
    # Calculate Donchian channels from previous 20 periods (use shift to avoid look-ahead)
    prev_high = np.concatenate([[high[0]], high[:-1]])  # shift(1)
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > 1d HMA21 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > hma_21_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND close < 1d HMA21 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < hma_21_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d HMA21 (trend reversal) OR touches Donchian lower (support)
            if close[i] < hma_21_1d_aligned[i] or close[i] <= donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d HMA21 (trend reversal) OR touches Donchian upper (resistance)
            if close[i] > hma_21_1d_aligned[i] or close[i] >= donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals