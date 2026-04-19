#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Trix + 1d Volume Spike + Choppiness Regime
# Trix (TRIple Exponential Average) filters noise and captures momentum
# Only trade when Trix crosses signal line with volume confirmation
# Use 1d choppiness regime to avoid whipsaw in sideways markets
# Designed to work in bull (momentum continuation) and bear (mean reversion in range)
# Target: 20-40 trades/year to minimize fee drag
name = "4h_Trix_Volume_Chop_1d_v1"
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
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Choppiness Index (CHOP) for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for CHOP
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_high_low = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low / atr_1d) / np.log10(14)
    chop = np.where(sum_high_low > 0, chop, 50)  # avoid division by zero
    chop_1d = chop
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Trix indicator (12-period EMA applied 3 times)
    # First EMA
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Second EMA of first EMA
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Third EMA of second EMA
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    # Trix = 100 * (ema3 - previous ema3) / previous ema3
    trix_raw = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0
    # Signal line: 9-period EMA of Trix
    trix_signal = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(chop_1d_aligned[i]) or \
           np.isnan(trix_raw[i]) or np.isnan(trix_signal[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 2.0x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 2.0 * avg_volume
        
        # Chop regime: CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
        chop_val = chop_1d_aligned[i]
        ranging_market = chop_val > 61.8
        trending_market = chop_val < 38.2
        
        if position == 0:
            # Long: Trix crosses above signal + volume + not in strong ranging market
            if trix_raw[i] > trix_signal[i] and trix_raw[i-1] <= trix_signal[i-1] and volume_filter and not ranging_market:
                signals[i] = 0.25
                position = 1
            # Short: Trix crosses below signal + volume + not in strong ranging market
            elif trix_raw[i] < trix_signal[i] and trix_raw[i-1] >= trix_signal[i-1] and volume_filter and not ranging_market:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Trix crosses below signal or ATR stop or strong ranging market
            if (trix_raw[i] < trix_signal[i] and trix_raw[i-1] >= trix_signal[i-1]) or \
               price < close[i-1] - 2.0 * atr or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Trix crosses above signal or ATR stop or strong ranging market
            if (trix_raw[i] > trix_signal[i] and trix_raw[i-1] <= trix_signal[i-1]) or \
               price > close[i-1] + 2.0 * atr or ranging_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals