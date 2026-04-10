#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high + 1d volume > 1.3x 20-period volume SMA + chop > 61.8 (ranging market)
# - Short when price breaks below Donchian(20) low + 1d volume > 1.3x 20-period volume SMA + chop > 61.8
# - Exit: price returns to Donchian(20) midpoint
# - Position sizing: 0.25 discrete level
# - Donchian breakout captures momentum, volume confirms participation, chop filter ensures mean reversion context
# - Works in bull/bear: breakouts work in trending markets, chop filter avoids false signals in strong trends
# - 4h timeframe targets 20-50 trades/year with strict entry conditions to minimize fee drag

name = "4h_1d_donchian_volume_chop_v1"
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
    
    # Calculate 4h Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h Choppiness Index(14)
    atr = pd.Series(np.maximum(high - low, 
                               np.maximum(np.abs(high - np.roll(close, 1)), 
                                          np.abs(low - np.roll(close, 1))))).rolling(window=14, min_periods=14).mean().values
    atr[0] = high[0] - low[0]
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr * 14 / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period SMA (volume spike)
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        vol_confirm = vol_1d_current[i] > 1.3 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (favorable for mean reversion after breakout)
        ranging_market = chop[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        return_to_mid = abs(close[i] - donchian_mid[i]) < (donchian_high[i] - donchian_low[i]) * 0.1  # Within 10% of midpoint
        
        # Entry conditions: Donchian breakout with volume and regime confirmation
        long_entry = breakout_up and vol_confirm and ranging_market
        short_entry = breakout_down and vol_confirm and ranging_market
        
        # Exit conditions: price returns to Donchian midpoint (mean reversion)
        long_exit = return_to_mid  # Exit long when price returns to midpoint
        short_exit = return_to_mid  # Exit short when price returns to midpoint
        
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
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals