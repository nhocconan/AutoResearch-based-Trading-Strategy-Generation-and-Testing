#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.8x 20-period average AND chop > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.8x 20-period average AND chop > 61.8 (range regime)
# - Exit when price returns to Camarilla Pivot level (mean reversion to mean)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# - Camarilla pivot levels work well in ranging markets (2025-2026 bear/range)
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we only trade in ranging regimes where mean reversion works

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 4h Camarilla pivot levels (based on previous day)
    # Need to shift by 1 to avoid look-ahead: use previous day's OHLC
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    # Previous day's OHLC (1 bar shift for 1d data)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = pivot + (range_hl * 1.1 / 4)
    l3 = pivot - (range_hl * 1.1 / 4)
    h4 = pivot + (range_hl * 1.1 / 2)
    l4 = pivot - (range_hl * 1.1 / 2)
    
    # Pre-compute 4h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # Pre-compute 1d Chopiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.zeros_like(tr)
    atr_1d[13] = np.mean(tr[1:14])  # First ATR value
    for i in range(14, len(tr)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Chopiness Index = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    sum_tr14 = rolling_sum(tr, 14)
    chop = 100 * np.log10(sum_tr14 / (atr_1d * 14)) / np.log10(14)
    chop_regime = chop > 61.8  # Range regime
    
    # Align HTF indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or np.isnan(pivot[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND range regime
            if (close[i] > h3[i] and 
                volume_spike_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND range regime
            elif (close[i] < l3[i] and 
                  volume_spike_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot level (mean reversion)
            exit_long = (position == 1 and close[i] < pivot[i])
            exit_short = (position == -1 and close[i] > pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals