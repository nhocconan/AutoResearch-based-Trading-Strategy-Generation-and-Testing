#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.3x 20-period average AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.3x 20-period average AND chop > 61.8
# - Exit when price returns to Camarilla Pivot level (mean reversion to center)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Camarilla levels work well in ranging markets which are common in 2025 bear/bias
# - Volume confirmation reduces false breakouts
# - Chop filter ensures we trade in ranging conditions where mean reversion works

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 12h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # Pre-compute 12h Chopiness Index (14-period) for regime filter
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # First bar uses current close
    tr = true_range(high, low, prev_close)
    
    # Calculate ATR (smoothed TR)
    atr = np.zeros_like(tr)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate Chopiness Index: 100 * log10(sum(ATR)/ (max(high)-min(low))) / log10(period)
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * (np.log10(sum_atr) - np.log10(range_hl)) / np.log10(14)
    chop_regime = chop > 61.8  # Chop > 61.8 indicates ranging market
    
    # Pre-compute 1d Camarilla levels from previous day
    # Camarilla: based on previous day's OHLC
    prev_close_1d = np.roll(df_1d['close'].values, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    
    # First day has no previous data
    prev_close_1d[0] = df_1d['close'].values[0]
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    # Calculate Camarilla levels
    range_1d = prev_high_1d - prev_low_1d
    camarilla_pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_h3 = camarilla_pivot + (range_1d * 1.1 / 4)
    camarilla_l3 = camarilla_pivot - (range_1d * 1.1 / 4)
    camarilla_h4 = camarilla_pivot + (range_1d * 1.1 / 2)
    camarilla_l4 = camarilla_pivot - (range_1d * 1.1 / 2)
    
    # Align HTF indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above H3 AND volume spike AND chop regime (ranging)
            if (close[i] > camarilla_h3_aligned[i] and 
                volume_spike_aligned[i] and 
                chop_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below L3 AND volume spike AND chop regime (ranging)
            elif (close[i] < camarilla_l3_aligned[i] and 
                  volume_spike_aligned[i] and 
                  chop_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to pivot level (mean reversion)
            exit_long = (position == 1 and close[i] < camarilla_pivot_aligned[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot_aligned[i])
            
            # Also exit if price breaks through H4/L4 (strong breakout against position)
            exit_long |= (position == 1 and close[i] > camarilla_h4_aligned[i])
            exit_short |= (position == -1 and close[i] < camarilla_l4_aligned[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals