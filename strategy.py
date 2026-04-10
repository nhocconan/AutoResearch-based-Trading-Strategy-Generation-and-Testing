#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.2x 20-bar average AND chop > 61.8 (ranging market)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.2x 20-bar average AND chop > 61.8
# - Exit when price returns to Donchian(20) midpoint or opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Donchian breakouts work in both bull and bear markets as momentum signals
# - Volume confirmation filters weak breakouts
# - Chop regime filter ensures we only trade in ranging markets where mean reversion works
# - Works on BTC/ETH/SOL as it's based on price structure and volume

name = "4h_1d_donchian_breakout_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    # Donchian high and low
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Pre-compute 1d volume confirmation: > 1.2x 20-period average
    volume_1d = df_1d['volume'].values
    volume_20_avg = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.2 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute chop regime filter on 1d data
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: chop > 61.8 indicates ranging market
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(1) - using TR as approximation for simplicity
    atr_1 = tr
    
    # Sum of ATR(1) over 14 periods
    sum_atr_14 = pd.Series(atr_1).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop calculation
    chop = np.where(
        (highest_high_14 - lowest_low_14) != 0,
        100 * np.log10(sum_atr_14 / (np.log10(14) * (highest_high_14 - lowest_low_14))),
        50  # Default when range is zero
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Chop regime: > 61.8 indicates ranging market (good for mean reversion breakouts)
    chop_regime = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and chop regime
            if (close_4h[i] > donchian_high[i] and 
                vol_spike_1d_aligned[i] and 
                chop_regime[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike and chop regime
            elif (close_4h[i] < donchian_low[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_regime[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian midpoint
            # 2. Opposite breakout occurs
            if position == 1:
                if close_4h[i] <= donchian_mid[i] or close_4h[i] < donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if close_4h[i] >= donchian_mid[i] or close_4h[i] > donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals