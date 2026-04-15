#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w HMA trend filter
# Long when price breaks above 1d Donchian high + volume > 1.5x avg + 1w HMA rising
# Short when price breaks below 1d Donchian low + volume > 1.5x avg + 1w HMA falling
# Uses 1d ATR for volume SMA calculation to avoid look-ahead
# Designed for low trade frequency (7-25/year) to minimize fee drag while capturing breakouts in trending markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian Channel (20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # === 1d Indicators: ATR for volume SMA calculation ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low (no previous close)
    tr[0] = high_1d[0] - low_1d[0]
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume SMA using ATR period for stability
    vol_sma = pd.Series(volume).rolling(window=14, min_periods=14).mean().values
    
    # === 1w Indicators: HMA(21) for trend filter ===
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA components
    wma_half = np.array([np.nan] * len(close_1w))
    wma_full = np.array([np.nan] * len(close_1w))
    
    for i in range(half_len, len(close_1w)):
        wma_half[i] = wma(close_1w[i-half_len+1:i+1], half_len)
    
    for i in range(21, len(close_1w)):
        wma_full[i] = wma(close_1w[i-21+1:i+1], 21)
    
    # HMA = 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final WMA of sqrt(n) on raw HMA
    hma_1w = np.array([np.nan] * len(close_1w))
    for i in range(sqrt_len, len(raw_hma)):
        if not np.isnan(raw_hma[i-sqrt_len+1:i+1]).any():
            hma_1w[i] = wma(raw_hma[i-sqrt_len+1:i+1], sqrt_len)
    
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # HMA slope (rising/falling)
    hma_slope = np.diff(hma_1w_aligned, prepend=hma_1w_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_sma[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 14-period volume SMA
        vol_confirm = volume[i] > (vol_sma[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d upper Donchian
        # 2. Volume confirmation
        # 3. 1w HMA rising (uptrend)
        if (close[i] > donchian_high_aligned[i]) and vol_confirm and hma_rising[i]:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d lower Donchian
        # 2. Volume confirmation
        # 3. 1w HMA falling (downtrend)
        elif (close[i] < donchian_low_aligned[i]) and vol_confirm and hma_falling[i]:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_Volume_HMA_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0