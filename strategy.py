#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with 4h volume spike and 1d chop regime filter
# - Long: price breaks above 4h H3 level + 4h volume > 1.5x 20-period MA + 1d CHOP(14) < 61.8 (trending regime)
# - Short: price breaks below 4h L3 level + 4h volume > 1.5x 20-period MA + 1d CHOP(14) < 61.8 (trending regime)
# - Exit: price returns to 4h Camarilla Pivot level or opposite signal
# - Position sizing: 0.20 (discrete level to minimize fee churn)
# - Session filter: 08-20 UTC to reduce noise trades
# - Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
# - Uses 4h for signal direction and structure, 1h only for entry timing precision
# - Volume confirmation ensures institutional participation, chop filter avoids ranging markets
# - Works in bull/bear: breakouts with trend in bull, mean reversion exits in bear ranges

name = "1h_4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1h OHLC
    open_1h = prices['open'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    close_1h = prices['close'].values
    volume_1h = prices['volume'].values
    
    # Pre-compute 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Pre-compute 1d data for chop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 4h (prior completed 4h bar)
    # Pivot = (H + L + C) / 3
    # H3 = C + (H - L) * 1.1 / 4
    # L3 = C - (H - L) * 1.1 / 4
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    h3_4h = close_4h + range_4h * 1.1 / 4.0
    l3_4h = close_4h - range_4h * 1.1 / 4.0
    
    # Align Camarilla levels to 1h (using prior completed 4h bar)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    
    # Calculate 4h volume moving average (20-period)
    volume_4h_series = pd.Series(volume_4h)
    volume_ma_20_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20_4h)
    
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
    
    # Pre-compute session filter (08-20 UTC)
    # open_time is already datetime64[ms], so we can use .hour directly
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period (need at least 50 for ATR14/CHOP)
        # Skip if any required data is invalid or outside session
        if (np.isnan(pivot_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_ma_aligned[i]) or
            np.isnan(chop_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Get current 1h OHLC
        close_price = close_1h[i]
        
        # Get aligned 4h data for current 1h bar (completed 4h bar)
        pivot_current = pivot_aligned[i]
        h3_current = h3_aligned[i]
        l3_current = l3_aligned[i]
        volume_ma_current = volume_ma_aligned[i]
        volume_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        chop_current = chop_aligned[i]
        
        # Volume spike condition: current 4h volume > 1.5x 20-period MA
        volume_spike = volume_4h_current > 1.5 * volume_ma_current
        
        # Trending regime condition: CHOP < 61.8 (below choppy threshold)
        trending_regime = chop_current < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above H3 + volume spike + trending regime
            if (close_price > h3_current and volume_spike and trending_regime):
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below L3 + volume spike + trending regime
            elif (close_price < l3_current and volume_spike and trending_regime):
                position = -1
                signals[i] = -0.20
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals