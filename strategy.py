#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
# - Long: price breaks above Donchian(20) high + 1d HMA(21) uptrend + volume > 1.5x 20-period average
# - Short: price breaks below Donchian(20) low + 1d HMA(21) downtrend + volume > 1.5x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 4h timeframe: targets 20-50 trades/year to avoid fee drag
# - Works in bull/bear markets: trend filter prevents counter-trend trades, breakouts capture momentum

name = "4h_1d_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d HMA(21) for trend filter
    close_1d = df_1d['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = np.array([wma(close_1d[i:i+half_len], half_len) if i+half_len <= len(close_1d) else np.nan 
                         for i in range(len(close_1d))])
    wma_full = np.array([wma(close_1d[i:i+21], 21) if i+21 <= len(close_1d) else np.nan 
                         for i in range(len(close_1d))])
    raw_hma = 2 * wma_half - wma_full
    hma_21 = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
    # Pad beginning with NaN
    hma_21_padded = np.full(len(close_1d), np.nan)
    hma_21_padded[half_len + sqrt_len - 1:len(hma_21)+half_len + sqrt_len - 1] = hma_21
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21_padded)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if low_4h[i] < donchian_low[i] or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if high_4h[i] > donchian_high[i] or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Long: price breaks above Donchian high + 1d HMA uptrend (close > HMA)
                if high_4h[i] > donchian_high[i] and close_4h[i] > hma_21_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low + 1d HMA downtrend (close < HMA)
                elif low_4h[i] < donchian_low[i] and close_4h[i] < hma_21_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals