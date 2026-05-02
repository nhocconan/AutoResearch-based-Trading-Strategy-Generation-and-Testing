#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation
# Uses 12h primary timeframe for lower trade frequency (target: 50-150 total trades over 4 years)
# Donchian channels provide clear trend-following structure proven effective on SOLUSDT
# 1d ATR regime filter (ATR(7)/ATR(30) < 1.2) identifies low-volatility periods for breakout trading
# Volume spike (1.8x 20-period average) confirms institutional participation
# Designed with tight entry conditions to minimize fee drag while maintaining edge in bull/bear markets
# Target: 75-125 total trades over 4 years (19-31/year) - within proven winning range for 12h

name = "12h_Donchian20_1dATR_Regime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d ATR(7) and ATR(30) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First bar
    tr2[0] = high_1d[0] - close_1d[0]  # First bar
    tr3[0] = low_1d[0] - close_1d[0]   # First bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_7 = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # ATR regime: low volatility when ATR(7)/ATR(30) < 1.2
    atr_ratio = np.where(atr_30 != 0, atr_7 / atr_30, 1.0)
    atr_regime_low = atr_ratio < 1.2
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime_low)
    
    # Calculate Donchian channels (20-period) on 12h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume spike (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and ATR calculations)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian high + low volatility regime + volume spike
            if close[i] > donchian_high[i] and atr_regime_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + low volatility regime + volume spike
            elif close[i] < donchian_low[i] and atr_regime_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian low (reversal signal)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian high (reversal signal)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals