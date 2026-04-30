#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA21 trend filter and volume confirmation.
# Long when price breaks above 4h Donchian upper band with 1d uptrend (close > 1d HMA21) and volume > 1.8x 20-bar avg.
# Short when price breaks below 4h Donchian lower band with 1d downtrend (close < 1d HMA21) and volume > 1.8x 20-bar avg.
# Exit on opposite Donchian band touch (mean reversion within the channel).
# Uses proven Donchian breakout structure with strict volume confirmation and 1d HMA21 trend filter to limit trades.
# 1d HMA21 provides smooth trend filter, reducing false signals in choppy markets and bear rallies.
# Timeframe: 4h, HTF: 1d as per experiment guidelines.

name = "4h_Donchian20_1dHMA21_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d HMA21 for trend filter
    close_1d = df_1d['close'].values
    # HMA formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    wma_half = np.full_like(close_1d, np.nan)
    wma_full = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= half_len:
        wma_half[half_len-1:] = wma(close_1d, half_len)
    if len(close_1d) >= 21:
        wma_full[20:] = wma(close_1d, 21)
    
    raw_hma = 2 * wma_half - wma_full
    hma_21_1d = np.full_like(close_1d, np.nan)
    if len(raw_hma) >= sqrt_len:
        hma_21_1d[sqrt_len-1:] = wma(raw_hma, sqrt_len)
    
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # 4h Donchian channels (20-period)
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_hma = hma_21_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, uptrend (close > 1d HMA21), volume spike
            if (curr_high > curr_upper and 
                curr_close > curr_hma and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, downtrend (close < 1d HMA21), volume spike
            elif (curr_low < curr_lower and 
                  curr_close < curr_hma and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Donchian lower band (mean reversion)
            if curr_low <= curr_lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Donchian upper band (mean reversion)
            if curr_high >= curr_upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals