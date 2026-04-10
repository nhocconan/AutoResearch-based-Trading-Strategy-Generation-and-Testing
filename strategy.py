#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume spike and chop regime filter
# - Long when price breaks above Donchian(20) high + 12h volume > 1.8x 20-period volume SMA + Chop(14) > 61.8 (range regime)
# - Short when price breaks below Donchian(20) low + 12h volume > 1.8x 20-period volume SMA + Chop(14) > 61.8
# - Exit: price returns to Donchian midpoint (mean reversion within range)
# - Position sizing: 0.30 discrete level
# - Donchian captures breakouts from ranges, volume confirms participation, chop filter ensures we trade in ranging markets where mean reversion works
# - Works in bull/bear: breakouts work in all regimes, chop filter avoids strong trends where breakouts fail
# - 4h timeframe targets 20-50 trades/year with strict entry conditions to minimize fee drag

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate 4h Chopiness Index (14-period) for regime filter
    # True Range
    tr1 = np.maximum(high - low, 
                     np.maximum(np.abs(high - np.roll(close, 1)), 
                                np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    # Sum of TR over period
    sum_tr = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Chop formula: 100 * log10(sum_TR / (HH - LL)) / log10(N)
    # Avoid division by zero and log of zero/negative
    hl_range = hh - ll
    chop = np.where((hl_range > 0) & (sum_tr > 0), 
                    100 * np.log10(sum_tr / hl_range) / np.log10(14), 
                    50)  # default to neutral when invalid
    
    # Calculate 12h volume SMA(20) for confirmation
    volume_12h = df_12h['volume'].values
    volume_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_sma_20_12h)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x 20-period SMA (volume spike)
        vol_12h_current = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)
        vol_confirm = vol_12h_current[i] > 1.8 * volume_sma_20_12h_aligned[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (favorable for mean reversion breakouts)
        ranging_market = chop[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > highest_high[i]  # Price breaks above upper band
        breakout_down = close[i] < lowest_low[i]  # Price breaks below lower band
        return_to_mid = abs(close[i] - donchian_mid[i]) < (highest_high[i] - lowest_low[i]) * 0.1  # Within 10% of midpoint
        
        # Entry conditions: Donchian breakout with volume and regime confirmation
        long_entry = breakout_up and vol_confirm and ranging_market
        short_entry = breakout_down and vol_confirm and ranging_market
        
        # Exit conditions: price returns to Donchian midpoint (mean reversion)
        long_exit = return_to_mid  # Exit long when price returns to midpoint
        short_exit = return_to_mid  # Exit short when price returns to midpoint
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.30
            elif short_entry:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals