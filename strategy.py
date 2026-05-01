#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Width regime + Donchian(20) breakout + volume confirmation
# BBW < 0.02 = low volatility squeeze (regime filter). Breakout from Donchian(20) with volume > 1.5x 20-bar mean.
# Long when price breaks above Donchian upper band in low BBW regime + volume spike.
# Short when price breaks below Donchian lower band in low BBW regime + volume spike.
# Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
# BBW regime filters out whipsaws in ranging markets, capturing only explosive moves after consolidation.

name = "6h_BBWRegime_Donchian20_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Band Width (20,2) regime filter
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid  # normalized width
    
    # Donchian channels (20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators (20 periods)
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(bb_width[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # BBW regime: low volatility squeeze (< 0.02)
        low_volatility_regime = bb_width[i] < 0.02
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high in low BBW regime + volume spike
            if curr_close > donch_high[i] and low_volatility_regime and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in low BBW regime + volume spike
            elif curr_close < donch_low[i] and low_volatility_regime and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below Donchian mid or BBW expands beyond threshold
            if curr_close < bb_mid[i] or bb_width[i] > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above Donchian mid or BBW expands beyond threshold
            if curr_close > bb_mid[i] or bb_width[i] > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals