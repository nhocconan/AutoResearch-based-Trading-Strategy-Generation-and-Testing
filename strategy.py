#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume regime filter (volume > 1.5x 20-day avg)
# Long when price breaks above Donchian upper channel in high volume regime
# Short when price breaks below Donchian lower channel in high volume regime
# Uses 1d volume regime to filter for institutional participation, avoiding low-volume breakouts
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag
# Works in bull/bear: volume confirms breakout strength regardless of trend direction

name = "4h_Donchian20_1dVolumeRegime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for daily calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume MA(20) for regime filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Volume regime: volume > 1.5x 20-day average (institutional participation)
    volume_regime_1d = vol_1d > (1.5 * vol_ma_20_1d)
    
    # Align daily volume regime to 4h timeframe (completed 1d bar only)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime_1d.astype(float))
    
    # Donchian(20) channels on 4h
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for Donchian and volume regime
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(volume_regime_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_volume_regime = volume_regime_aligned[i] > 0.5  # boolean from aligned float
        
        if position == 0:  # Flat - look for new entries
            # Only trade in high volume regime (institutional participation)
            if curr_volume_regime:
                # Bullish breakout: price breaks above upper Donchian channel
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price breaks below lower Donchian channel
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price returns to middle of channel
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close <= middle_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price returns to middle of channel
            middle_channel = (curr_upper + curr_lower) / 2.0
            
            if curr_close >= middle_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals