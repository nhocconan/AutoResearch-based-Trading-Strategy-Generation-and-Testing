#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume regime filter (volume > 1.5x 20d avg) and ATR-based trailing stop
# Long when price breaks above Donchian upper channel in high-volume regime
# Short when price breaks below Donchian lower channel in high-volume regime
# Uses 1d volume to filter for institutional participation, avoiding low-volume breakouts
# ATR trailing stop (2.5x ATR) to protect gains and limit drawdown
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag

name = "4h_Donchian20_1dVolumeRegime_ATRStop_v1"
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
    
    # Calculate 1d 20-period volume average for regime filter
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_regime = df_1d['volume'].values > (1.5 * vol_ma_20_1d)
    
    # Align daily volume regime to 4h timeframe (completed 1d bar only)
    volume_regime_aligned = align_htf_to_ltf(prices, df_1d, volume_regime)
    
    # Donchian(20) channels on 4h
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # ATR(14) for trailing stop
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0  # for long trailing stop
    lowest_low_since_entry = 0    # for short trailing stop
    
    start_idx = max(35, 20)  # warmup for ATR and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(volume_regime_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_volume_regime = volume_regime_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade in high-volume regime (institutional participation)
            if curr_volume_regime:
                # Bullish breakout: price breaks above upper Donchian channel
                if curr_close > curr_upper:
                    signals[i] = 0.25
                    position = 1
                    highest_high_since_entry = curr_high
                # Bearish breakout: price breaks below lower Donchian channel
                elif curr_close < curr_lower:
                    signals[i] = -0.25
                    position = -1
                    lowest_low_since_entry = curr_low
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_high > highest_high_since_entry:
                highest_high_since_entry = curr_high
            
            # ATR trailing stop: exit if price drops 2.5x ATR from highest high
            if curr_close < (highest_high_since_entry - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_low < lowest_low_since_entry:
                lowest_low_since_entry = curr_low
            
            # ATR trailing stop: exit if price rises 2.5x ATR from lowest low
            if curr_close > (lowest_low_since_entry + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals