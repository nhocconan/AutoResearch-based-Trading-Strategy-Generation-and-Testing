#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long: price breaks above Camarilla H3 level + 1d volume > 1.5x 20-period MA + CHOP(14) < 61.8 (trending regime)
# - Short: price breaks below Camarilla L3 level + 1d volume > 1.5x 20-period MA + CHOP(14) < 61.8 (trending regime)
# - Exit: price returns to Camarilla Pivot level or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 80-160 total trades over 4 years (20-40/year) to avoid fee drag
# - Camarilla levels provide intraday support/resistance, volume confirms institutional participation,
#   chop filter avoids false breakouts in ranging markets. Works in bull/bear: breakouts with trend in bull,
#   mean reversion exits in bear ranges.

name = "4h_1d_camarilla_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 4h OHLC
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Pivot = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1 / 4
    # L3 = C - (H - L) * 1.1 / 4
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = close_1d + range_1d * 1.1 / 4.0
    l3_1d = close_1d - range_1d * 1.1 / 4.0
    
    # Align Camarilla levels to 4h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate Choppiness Index (14-period) for 1d
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (14 * log10(14))) / log10(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop_1d = 100 * np.log10(atr_sum / (14 * np.log10(14))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for ATR14/CHOP)
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h OHLC
        close_price = close_4h[i]
        
        # Get aligned 1d data for current 4h bar (completed 1d bar)
        pivot_current = pivot_aligned[i]
        h3_current = h3_aligned[i]
        l3_current = l3_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        chop_current = chop_aligned[i]
        
        # Volume spike condition: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d_current > 1.5 * volume_ma_current
        
        # Trending regime condition: CHOP < 61.8 (below choppy threshold)
        trending_regime = chop_current < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above H3 + volume spike + trending regime
            if (close_price > h3_current and volume_spike and trending_regime):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below L3 + volume spike + trending regime
            elif (close_price < l3_current and volume_spike and trending_regime):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to pivot level or opposite signal
            if position == 1 and close_price <= pivot_current:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close_price >= pivot_current:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals