#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with 1w volume spike and choppiness regime filter
# - Long: price breaks above Camarilla H3 level (from prior 1w) + 1w volume > 1.8x 20-period MA + CHOP(14) < 61.8 (trending regime)
# - Short: price breaks below Camarilla L3 level (from prior 1w) + 1w volume > 1.8x 20-period MA + CHOP(14) < 61.8 (trending regime)
# - Exit: price returns to Camarilla Pivot level (from prior 1w) or opposite signal
# - Position sizing: 0.25 (discrete level)
# - Target: 50-100 total trades over 4 years (12-25/year) to avoid fee drag
# - Using 1w HTF for Camarilla levels provides more significant support/resistance than 1d, reducing false breakouts
# - Volume confirmation on 1w ensures institutional participation, chop filter avoids ranging markets
# - Works in bull/bear: breakouts with trend in bull, mean reversion exits in bear ranges

name = "1d_1w_camarilla_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d OHLC
    open_1d = prices['open'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Camarilla pivot levels for 1w (prior completed week)
    # Pivot = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1 / 4
    # L3 = C - (H - L) * 1.1 / 4
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    h3_1w = close_1w + range_1w * 1.1 / 4.0
    l3_1w = close_1w - range_1w * 1.1 / 4.0
    
    # Align Camarilla levels to 1d (using prior completed 1w bar)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Calculate 1w volume moving average (20-period)
    volume_1w_series = pd.Series(volume_1w)
    volume_ma_20_1w = volume_1w_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate Choppiness Index (14-period) for 1w
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (14 * log10(14))) / log10(14)
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop_1w = 100 * np.log10(atr_sum / (14 * np.log10(14))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for ATR14/CHOP)
        # Skip if any required data is invalid
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d OHLC
        close_price = close_1d[i]
        
        # Get aligned 1w data for current 1d bar (completed 1w bar)
        pivot_current = pivot_aligned[i]
        h3_current = h3_aligned[i]
        l3_current = l3_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        chop_current = chop_aligned[i]
        
        # Volume spike condition: current 1w volume > 1.8x 20-period MA
        volume_spike = volume_1w_current > 1.8 * volume_ma_current
        
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