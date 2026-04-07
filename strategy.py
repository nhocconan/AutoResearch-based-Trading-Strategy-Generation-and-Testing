#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w volatility regime filter
# Uses Donchian(20) breakout for trend entry, confirmed by 1d volume spike and filtered by 1w ATR-based volatility regime.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at channel extremes.

name = "4h_donchian20_1d_volume_1w_vol_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly data for volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily volume spike (volume > 1.5 * 20-day average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Weekly ATR-based volatility regime (low volatility = trending, high volatility = range)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1w = pd.Series(atr_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    vol_ratio_1w = atr_1w / atr_ma_1w
    vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    
    # Volatility regime: low vol (ratio < 0.8) = trend following, high vol (ratio > 1.2) = mean reversion
    low_vol = vol_ratio_1w_aligned < 0.8
    high_vol = vol_ratio_1w_aligned > 1.2
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Base position size
        base_size = 0.25
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high[i-1]  # break above previous high
        breakout_down = close[i] < donch_low[i-1]  # break below previous low
        
        # Low volatility regime: trend following
        if low_vol[i]:
            if breakout_up and vol_spike_1d_aligned[i]:
                signals[i] = base_size  # long
            elif breakout_down and vol_spike_1d_aligned[i]:
                signals[i] = -base_size  # short
        
        # High volatility regime: mean reversion at channel extremes
        elif high_vol[i]:
            # Near upper channel: short
            if close[i] >= 0.95 * donch_high[i]:
                signals[i] = -base_size
            # Near lower channel: long
            elif close[i] <= 1.05 * donch_low[i]:
                signals[i] = base_size
    
    return signals