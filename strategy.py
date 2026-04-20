#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h trend following with 4h ADX filter and 1d volume spike
# Uses 4h ADX > 25 to identify trending markets (avoids chop)
# Uses 1d volume > 2x 20-day average to confirm institutional interest
# Enters long when price > 4h EMA20 and short when price < 4h EMA20
# Designed to work in both bull and bear markets by only trading strong trends
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for trend filter (ADX and EMA)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ADX(14)
    plus_dm = np.zeros_like(high_4h)
    minus_dm = np.zeros_like(low_4h)
    for i in range(1, len(high_4h)):
        up = high_4h[i] - high_4h[i-1]
        down = low_4h[i-1] - low_4h[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    tr = np.maximum(high_4h - low_4h, 
                    np.maximum(np.abs(high_4h - np.roll(high_4h, 1)), 
                               np.abs(low_4h - np.roll(low_4h, 1))))
    tr[0] = high_4h[0] - low_4h[0]
    
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    plus_di = 100 * np.where(atr != 0, 
                             np.convolve(plus_dm, np.ones(14)/14, mode='full')[:len(plus_dm)] / atr, 0)
    minus_di = 100 * np.where(atr != 0, 
                              np.convolve(minus_dm, np.ones(14)/14, mode='full')[:len(minus_dm)] / atr, 0)
    dx = 100 * np.where((plus_di + minus_di) != 0, 
                        np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = np.convolve(dx, np.ones(14)/14, mode='full')[:len(dx)]
    adx[:13] = np.nan
    
    # Calculate 4h EMA20
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h indicators to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Load 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # 1h data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in indicators
        if np.isnan(adx_4h_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend and volume conditions
        is_trending = adx_4h_aligned[i] > 25
        has_volume_spike = vol_spike_1d_aligned[i] > 0.5
        price_above_ema = close[i] > ema20_4h_aligned[i]
        price_below_ema = close[i] < ema20_4h_aligned[i]
        
        if position == 0:
            # Enter long in uptrend with volume spike
            if is_trending and has_volume_spike and price_above_ema:
                signals[i] = 0.20
                position = 1
            # Enter short in downtrend with volume spike
            elif is_trending and has_volume_spike and price_below_ema:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend weakness or price below EMA
            if not (is_trending and price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend weakness or price above EMA
            if not (is_trending and price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ADX_TrendFilter_VolumeSpike"
timeframe = "1h"
leverage = 1.0