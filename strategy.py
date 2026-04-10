#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period average AND choppiness index < 61.8 (trending market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period average AND choppiness index < 61.8
# - Exit when price returns to Camarilla Pivot point (mean reversion to equilibrium)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Camarilla levels provide precise intraday support/resistance with statistical edge
# - Volume confirmation reduces false breakouts
# - Choppiness filter avoids whipsaws in ranging markets

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
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1d choppiness index (14-period)
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
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sumTR / (HH - LL)) / log10(14)
    # Avoid division by zero
    hl_range = hh_14 - ll_14
    chop = np.full_like(sum_tr, 50.0, dtype=float)  # Default to neutral
    mask = (hl_range > 0) & (~np.isnan(sum_tr))
    chop[mask] = 100 * np.log10(sum_tr[mask] / hl_range[mask]) / np.log10(14)
    
    # Trending regime: CHOP < 61.8
    trending_regime = chop < 61.8
    
    # Align HTF indicators to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime)
    
    # Pre-compute 12h Camarilla levels (based on previous day's OHLC)
    # We need to align 1d OHLC to 12h timeframe
    df_1d_index = pd.DatetimeIndex(df_1d.index)
    
    # Create arrays for 1d OHLC aligned to 12h
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['open'].values)
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['high'].values)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['low'].values)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    # Calculate Camarilla levels for each 12h bar (using previous 1d close)
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4, L4 = close - 1.1*(high-low)/2
    # Pivot = (high + low + close)/3
    hl_range_1d = high_1d_aligned - low_1d_aligned
    camarilla_pivot = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    camarilla_h3 = camarilla_pivot + 1.1 * hl_range_1d / 4.0
    camarilla_l3 = camarilla_pivot - 1.1 * hl_range_1d / 4.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(trending_regime_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: price breaks above Camarilla H3 AND volume spike AND trending regime
            if (close[i] > camarilla_h3[i] and 
                volume_spike_aligned[i] and 
                trending_regime_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: price breaks below Camarilla L3 AND volume spike AND trending regime
            elif (close[i] < camarilla_l3[i] and 
                  volume_spike_aligned[i] and 
                  trending_regime_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: price returns to Camarilla Pivot point (mean reversion)
            exit_long = (position == 1 and close[i] < camarilla_pivot[i])
            exit_short = (position == -1 and close[i] > camarilla_pivot[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals