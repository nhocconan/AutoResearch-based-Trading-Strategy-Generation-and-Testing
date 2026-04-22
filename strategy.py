#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d Donchian(20) breakout with volume confirmation.
# Long when price breaks above Donchian(20) high in trending regime (CHOP < 38.2) with volume spike.
# Short when price breaks below Donchian(20) low in trending regime (CHOP < 38.2) with volume spike.
# Uses daily Donchian for structure, 12h Choppiness for regime filter, volume spike for confirmation.
# Designed for 12h timeframe to target 15-35 trades/year per symbol.
# Works in bull/bear via regime filter - only trades in trending markets, avoids whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian channels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Load 12h data for Choppiness Index (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 1d Donchian(20) channels
    # Upper = max(high_1d, 20 periods)
    # Lower = min(low_1d, 20 periods)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h Choppiness Index (14-period)
    # ATR(14) = sum(|high-low|) / 14 over 14 periods
    # Chop = 100 * log10(ATR(14) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    # Avoid division by zero
    chop_raw = np.where(range_14 > 0, atr14 / range_14, 1.0)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Align to 12h timeframe (waits for 1d/12h bar to close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume spike filter (25-period for 12h)
    vol_ma25 = pd.Series(volume).rolling(window=25, min_periods=25).mean().values
    vol_spike = volume > 2.5 * vol_ma25  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma25[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0 and is_trending:
            # Long: price breaks above Donchian high + volume spike
            if (close[i] > donch_high_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike
            elif (close[i] < donch_low_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on Donchian touch or regime change to ranging
                if (close[i] < donch_low_aligned[i] or chop_aligned[i] >= 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Donchian touch or regime change to ranging
                if (close[i] > donch_high_aligned[i] or chop_aligned[i] >= 38.2):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_ChoppinessRegime_Donchian20_Breakout_VolumeSpike"
timeframe = "12h"
leverage = 1.0