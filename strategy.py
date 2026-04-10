#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter
# - Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-bar average AND 1d choppiness < 61.8 (trending)
# - Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-bar average AND 1d choppiness < 61.8 (trending)
# - Exit when price returns to Donchian(20) midpoint or opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong trending moves; volume confirmation filters false breakouts
# - Choppiness regime filter ensures we only trade in trending markets (avoids whipsaws in ranging markets)
# - Works in both bull and bear markets by trading breakouts in the direction of the trend

name = "12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels on 12h data (primary timeframe)
    donchian_period = 20
    high_12h = prices['high'].rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_12h = prices['low'].rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (high_12h + low_12h) / 2.0
    
    # Pre-compute 1d volume confirmation: > 1.5x 20-period average
    volume_20_avg = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'] > (1.5 * volume_20_avg)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.values)
    
    # Pre-compute 1d choppiness regime filter: CHOP < 61.8 = trending (good for breakouts)
    chop_period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of True Range over chop_period
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Highest high and lowest low over chop_period
    hh_period = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    ll_period = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Choppiness Index = 100 * log10(atr_sum / (hh_period - ll_period)) / log10(chop_period)
    # CHOP > 61.8 = ranging, CHOP < 38.2 = strong trending
    range_diff = hh_period - ll_period
    choppiness = np.where(
        (range_diff > 0) & (atr_sum > 0),
        100 * np.log10(atr_sum / range_diff) / np.log10(chop_period),
        50  # Default when range is zero
    )
    chop_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and trending regime
            if (prices['close'].iloc[i] > high_12h[i] and 
                vol_spike_1d_aligned[i] and 
                chop_aligned[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike and trending regime
            elif (prices['close'].iloc[i] < low_12h[i] and 
                  vol_spike_1d_aligned[i] and 
                  chop_aligned[i] < 61.8):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian midpoint (mean reversion)
            # 2. Opposite breakout occurs (long exits on downside breakout, short exits on upside breakout)
            if position == 1:
                if (prices['close'].iloc[i] <= donchian_mid[i] or 
                    prices['close'].iloc[i] < low_12h[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if (prices['close'].iloc[i] >= donchian_mid[i] or 
                    prices['close'].iloc[i] > high_12h[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals