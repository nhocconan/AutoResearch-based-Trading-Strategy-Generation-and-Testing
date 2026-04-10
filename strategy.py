#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot structure and volume confirmation
# - Long when price breaks above Donchian(20) high + price > 1d Camarilla H3 level + volume > 1.5x 20-period 6h volume SMA
# - Short when price breaks below Donchian(20) low + price < 1d Camarilla L3 level + volume > 1.5x 20-period 6h volume SMA
# - Exit: price returns to Donchian(20) midpoint
# - Position sizing: 0.25 discrete level
# - Donchian breakout captures institutional momentum
# - Camarilla H3/L3 levels act as institutional support/resistance from 1d timeframe
# - Volume confirmation ensures breakout validity
# - Works in bull/bear: breakouts occur in all regimes, Camarilla levels adapt to volatility

name = "6h_1d_donchian_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate Donchian Channel on primary timeframe (6h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 6h volume SMA for confirmation (20-period)
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4, etc.
    # We use: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4.0
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_sma_20[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period SMA
        vol_confirm = volume[i] > 1.5 * volume_sma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]  # Price breaks above Donchian high
        breakout_down = close[i] < lowest_low[i]   # Price breaks below Donchian low
        
        # Camarilla level conditions
        above_h3 = close[i] > camarilla_h3_aligned[i]  # Price above Camarilla H3
        below_l3 = close[i] < camarilla_l3_aligned[i]  # Price below Camarilla L3
        
        # Entry conditions
        long_entry = breakout_up and above_h3 and vol_confirm
        short_entry = breakout_down and below_l3 and vol_confirm
        
        # Exit condition: price returns to Donchian midpoint
        exit_long = close[i] < donchian_mid[i]
        exit_short = close[i] > donchian_mid[i]
        
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