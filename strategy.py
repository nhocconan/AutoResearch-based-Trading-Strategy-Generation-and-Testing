#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with HMA(21) trend filter and volume confirmation
# Donchian breakouts capture momentum shifts with clear structure. HMA(21) provides smooth trend
# direction without excessive lag. Volume confirmation ensures institutional participation.
# Designed for 4h timeframe with 1d HTF for trend alignment, targeting 20-40 trades/year
# to minimize fee drag while maintaining edge in both bull and bear markets via trend following.

name = "4h_Donchian20_HMA21_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for HTF trend and structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d HMA(21) for trend filter
    close_1d = df_1d['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Pad arrays for WMA calculation
    wma_half = np.full_like(close_1d, np.nan)
    wma_full = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= half_len:
        wma_half[half_len-1:] = wma(close_1d, half_len)
    if len(close_1d) >= 21:
        wma_full[20:] = wma(close_1d, 21)
    
    raw_hma = 2 * wma_half - wma_full
    hma_21 = np.full_like(close_1d, np.nan)
    if len(raw_hma) >= sqrt_len:
        hma_21[sqrt_len-1:] = wma(raw_hma[sqrt_len-1:], sqrt_len)
    
    # Align 1d HMA to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i]) if i >= 20 else np.mean(volume[:i]) if i > 0 else 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high_4h = high_4h[i]
        curr_low_4h = low_4h[i]
        curr_hma = hma_21_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: price breaks above 4h Donchian upper with 1d uptrend
                if curr_close > curr_high_4h and curr_close > curr_hma:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 4h Donchian lower with 1d downtrend
                elif curr_close < curr_low_4h and curr_close < curr_hma:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 4h Donchian lower
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_low_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 4h Donchian upper
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_high_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals