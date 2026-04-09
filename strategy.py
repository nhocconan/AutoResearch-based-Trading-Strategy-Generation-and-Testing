#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d volume regime + ATR trailing stop
# - Primary signal: Donchian(20) breakout on 4h timeframe (price > 20-period high for long, < 20-period low for short)
# - Regime filter: 1d volume > 50-period median volume (avoid low-participation periods)
# - Exit: ATR(14) trailing stop (3 * ATR from extreme price)
# - Position size: 0.30 (discrete level) to balance risk and return
# - Works in bull/bear: Donchian captures trends, volume filter ensures participation, ATR stop manages risk in volatile markets
# - Target: 20-50 trades/year (80-200 total over 4 years) per 4h strategy guidelines

name = "4h_1d_donchian_volume_atr_v1"
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
    
    # Pre-compute 1d volume regime
    volume_1d = df_1d['volume'].values
    median_volume_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).median().values
    volume_regime_1d = volume_1d > median_volume_50
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR(14) for trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_stop = 0.0
    short_stop = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Update trailing stop
            long_stop = max(long_stop, high[i] - 3.0 * atr[i])
            # Exit: price hits trailing stop OR price closes below Donchian low
            if close[i] <= long_stop or close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
                long_stop = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update trailing stop
            short_stop = min(short_stop, low[i] + 3.0 * atr[i])
            # Exit: price hits trailing stop OR price closes above Donchian high
            if close[i] >= short_stop or close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
                short_stop = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Look for Donchian breakout with volume regime confirmation
            # Long: price breaks above Donchian high AND volume regime
            if close[i] > highest_high[i] and volume_regime_aligned[i]:
                position = 1
                signals[i] = 0.30
                long_stop = high[i] - 3.0 * atr[i]
            # Short: price breaks below Donchian low AND volume regime
            elif close[i] < lowest_low[i] and volume_regime_aligned[i]:
                position = -1
                signals[i] = -0.30
                short_stop = low[i] + 3.0 * atr[i]
    
    return signals