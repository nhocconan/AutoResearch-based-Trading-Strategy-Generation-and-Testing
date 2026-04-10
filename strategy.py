#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# - Long when close breaks above Donchian(20) high + ATR(14) > ATR(50) (expanding volatility) + volume > 1.5x 20-period 1d volume SMA
# - Short when close breaks below Donchian(20) low + ATR(14) > ATR(50) + volume > 1.5x 20-period 1d volume SMA
# - Exit: close crosses Donchian midpoint (mean reversion)
# - Position sizing: 0.25 discrete level
# - Donchian breakouts capture momentum in both bull/bear markets
# - ATR expansion filter ensures we trade during volatile breakouts, not chop
# - Volume confirmation confirms institutional participation
# - Works in bull/bear: breakouts occur in all regimes, volume confirms validity

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Donchian channels on primary timeframe (4h)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate ATR on primary timeframe (4h)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = high[0] - low[0]  # first bar
    tr3[0] = high[0] - low[0]  # first bar
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate longer period ATR for filter (50-period)
    atr_long_period = 50
    atr_long = pd.Series(tr).rolling(window=atr_long_period, min_periods=atr_long_period).mean().values
    
    # Calculate 1d volume SMA for confirmation (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Align 1d volume to 4h timeframe
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_long[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period 1d volume SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # ATR filter: short-term ATR > long-term ATR (expanding volatility)
        atr_expanding = atr[i] > atr_long[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions: close crosses Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
        # Entry conditions: breakout + volume confirmation + ATR expansion
        long_entry = long_breakout and vol_confirm and atr_expanding
        short_entry = short_breakout and vol_confirm and atr_expanding
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals