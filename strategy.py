#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with 12h uptrend (price > 12h HMA21) and volume > 2.0x 20-bar avg.
# Short when price breaks below Donchian lower band with 12h downtrend (price < 12h HMA21) and volume > 2.0x 20-bar avg.
# Exit on opposite Donchian band touch (mean reversion within the channel).
# Uses HMA21 for smoother trend than EMA, reducing whipsaw in chop. Strict volume spike (2.0x) limits trades to target 19-50/year.
# 12h HMA21 provides intermediate trend filter between 4h and 1d for BTC/ETH, improving bear market performance.
# Timeframe: 4h, HTF: 12h as per experiment guidelines.

name = "4h_Donchian20_12hHMA21_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h HMA21 for trend filter: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='full')[-len(values):] / weights.sum()
    
    def hma(values, window):
        half = window // 2
        sqrt_n = int(np.sqrt(window))
        wma_half = wma(values, half)
        wma_full = wma(values, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_n)
    
    close_12h = df_12h['close'].values
    hma_21_12h = hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Previous 12h OHLC for completed 12h bar (no look-ahead)
    df_12h_prev = get_htf_data(prices, '12h')
    if len(df_12h_prev) < 2:
        return np.zeros(n)
    
    prev_high_12h = df_12h_prev['high'].shift(1).values
    prev_low_12h = df_12h_prev['low'].shift(1).values
    prev_close_12h = df_12h_prev['close'].shift(1).values
    
    # Align 12h data to 4h timeframe (completed 12h bar only)
    prev_high_aligned = align_htf_to_ltf(prices, df_12h_prev, prev_high_12h)
    prev_low_aligned = align_htf_to_ltf(prices, df_12h_prev, prev_low_12h)
    prev_close_aligned = align_htf_to_ltf(prices, df_12h_prev, prev_close_12h)
    
    # Donchian(20) from previous completed 12h bar (no look-ahead)
    donchian_high = pd.Series(prev_high_aligned).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low_aligned).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for HMA21 and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(hma_21_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_donch_high = donchian_high[i]
        curr_donch_low = donchian_low[i]
        curr_hma_21_12h = hma_21_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high, uptrend (price > 12h HMA21), volume spike
            if (curr_close > curr_donch_high and 
                curr_close > curr_hma_21_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, downtrend (price < 12h HMA21), volume spike
            elif (curr_close < curr_donch_low and 
                  curr_close < curr_hma_21_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: price touches Donchian low (mean reversion)
            if curr_close <= curr_donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: price touches Donchian high (mean reversion)
            if curr_close >= curr_donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals