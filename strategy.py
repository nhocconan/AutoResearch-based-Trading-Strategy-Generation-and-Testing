#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Donchian upper band (20-period high) + 1d volume > 2.0x 20-period volume SMA + Chop(14) > 61.8 (range regime)
# - Short when price breaks below Donchian lower band (20-period low) + 1d volume > 2.0x 20-period volume SMA + Chop(14) > 61.8
# - Exit: price returns to Donchian midpoint (mean reversion within channel)
# - Position sizing: 0.25 discrete level
# - Donchian channels provide structure in ranging markets
# - Volume confirms institutional participation, chop filter ensures we trade in ranging markets
# - Works in bull/bear: mean reversion at channel midpoint works in all regimes, chop filter avoids strong trends
# - 4h timeframe targets 19-50 trades/year with strict entry conditions to minimize fee drag

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
    
    # Calculate 4h Donchian Channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
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
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Align Donchian levels to 4h timeframe (using completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)  # Already 4h data
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)  # Already 4h data
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)      # Already 4h data
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (volume spike)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (favorable for mean reversion at channel midpoint)
        ranging_market = chop[i] > 61.8
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper_aligned[i]   # Price breaks above upper band
        breakout_down = close[i] < donchian_lower_aligned[i] # Price breaks below lower band
        return_to_mid = abs(close[i] - donchian_mid_aligned[i]) < (donchian_upper_aligned[i] - donchian_lower_aligned[i]) * 0.2  # Within 20% of midpoint
        
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