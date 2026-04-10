#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# - Long when price breaks above Camarilla H3 level + volume > 2.0x 20-period 1d volume SMA + CHOP(14) < 40 (trending)
# - Short when price breaks below Camarilla L3 level + volume > 2.0x 20-period 1d volume SMA + CHOP(14) < 40 (trending)
# - Exit: price returns to Camarilla pivot point (mean reversion)
# - Position sizing: 0.25 discrete level
# - Camarilla levels derived from 1d OHLC provide institutional support/resistance
# - Volume spike confirms institutional participation
# - Choppiness filter ensures we only trade in trending markets to avoid false breakouts
# - Works in bull/bear: breakouts occur in all regimes, CHOP filter prevents chop whipsaws

name = "12h_1d_camarilla_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels on 1d timeframe
    # Camarilla levels: based on previous day's OHLC
    # H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.125*(high-low), etc.
    # Pivot = (high + low + close) / 3
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    h3 = prev_close + 1.125 * range_hl  # Strong resistance
    l3 = prev_close - 1.125 * range_hl  # Strong support
    h4 = prev_close + 1.5 * range_hl    # Ultimate resistance
    l4 = prev_close - 1.5 * range_hl    # Ultimate support
    
    # Align Camarilla levels to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3.values)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3.values)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate Choppiness Index on 1d timeframe (14-period)
    # CHOP = 100 * log10(sum(ATR)/ (n * (max(high)-min(low)))) / log10(n)
    # Lower CHOP = trending, Higher CHOP = ranging
    # We want CHOP < 40 for trending markets
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d.iloc[0] = df_1d['high'].iloc[0] - df_1d['low'].iloc[0]  # first bar
    
    # Sum of ATR over 14 periods
    atr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max high and min low over 14 periods
    max_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop_1d = np.where(
        (max_high_14 - min_low_14) != 0,
        100 * np.log10(atr_sum_14 / (14 * (max_high_14 - min_low_14))) / np.log10(14),
        50  # default to neutral when range is zero
    )
    
    # Align Choppiness Index to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for confirmation (aligned to 12h)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: CHOP < 40 indicates trending market (lower = more trending)
        regime_filter = chop_1d_aligned[i] < 40
        
        # Camarilla breakout entry conditions
        # Long: price breaks above H3 level + volume confirmation + trending regime
        # Short: price breaks below L3 level + volume confirmation + trending regime
        long_entry = (close[i] > h3_aligned[i] and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (close[i] < l3_aligned[i] and 
                      vol_confirm and 
                      regime_filter)
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        exit_long = close[i] < pivot_aligned[i]
        exit_short = close[i] > pivot_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals