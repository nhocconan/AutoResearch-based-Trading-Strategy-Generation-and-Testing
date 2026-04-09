#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d HMA trend filter and volume confirmation
# - Long when price breaks above 4h Donchian high AND 1d HMA(21) is rising (uptrend)
# - Short when price breaks below 4h Donchian low AND 1d HMA(21) is falling (downtrend)
# - Volume confirmation: 1d volume > 1.5x 20-period average to ensure institutional participation
# - ATR-based stoploss (2.0x ATR) and position size 0.25 to limit drawdown
# - Designed for fewer trades (~30-50/year) to minimize fee drag while capturing strong trends
# - Works in bull markets (breakouts with rising HMA) and bear markets (breakdowns with falling HMA)

name = "4h_1d_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d True Range for ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # 1d ATR(14) for stoploss
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d HMA(21) for trend direction
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate HMA manually for compatibility
    wma_half = pd.Series(close_1d).rolling(window=half_len, min_periods=half_len).apply(
        lambda x: np.dot(x, np.arange(1, half_len+1)) / np.arange(1, half_len+1).sum(), raw=True
    ).values
    wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).apply(
        lambda x: np.dot(x, np.arange(1, 22)) / np.arange(1, 22).sum(), raw=True
    ).values
    
    hma_raw = 2 * wma_half - wma_full
    hma_1d = pd.Series(hma_raw).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
        lambda x: np.dot(x, np.arange(1, sqrt_len+1)) / np.arange(1, sqrt_len+1).sum(), raw=True
    ).values
    
    # 1d HMA rising/falling
    hma_rising = np.zeros_like(hma_1d, dtype=bool)
    hma_falling = np.zeros_like(hma_1d, dtype=bool)
    hma_rising[1:] = hma_1d[1:] > hma_1d[:-1]
    hma_falling[1:] = hma_1d[1:] < hma_1d[:-1]
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d Volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * avg_volume_20)
    
    # Align all 1d indicators to 4h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    hma_rising_aligned = align_htf_to_ltf(prices, df_1d, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_1d, hma_falling.astype(float))
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(hma_rising_aligned[i]) or np.isnan(hma_falling_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: price below Donchian low or ATR stoploss
            if low[i] <= donchian_low_aligned[i]:  # Price below lower band
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price above Donchian high or ATR stoploss
            if high[i] >= donchian_high_aligned[i]:  # Price above upper band
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with HMA trend filter and volume confirmation
            if (high[i] >= donchian_high_aligned[i] and  # Break above upper band
                hma_rising_aligned[i] and              # 1d HMA rising (uptrend)
                volume_spike_aligned[i]):              # Volume confirmation
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            elif (low[i] <= donchian_low_aligned[i] and    # Break below lower band
                  hma_falling_aligned[i] and             # 1d HMA falling (downtrend)
                  volume_spike_aligned[i]):              # Volume confirmation
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals