#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d volume spike and chop regime filter
# - Long when Williams %R < -80 (oversold) + 1d volume > 1.5x 20-period volume SMA + Chop(14) > 61.8 (range regime)
# - Short when Williams %R > -20 (overbought) + 1d volume > 1.5x 20-period volume SMA + Chop(14) > 61.8
# - Exit: Williams %R crosses back above -50 (for long) or below -50 (for short)
# - Position sizing: 0.25 discrete level
# - Williams %R identifies overextended moves in ranging markets where mean reversion occurs
# - Volume confirms participation, chop filter avoids strong trends
# - 4h timeframe targets 19-50 trades/year with strict entry conditions to minimize fee drag

name = "4h_1d_williamsr_volume_chop_v1"
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
    
    # Calculate 4h Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                          -50)  # default to neutral when range is zero
    
    # Calculate 1d OHLC for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        
        # Volume confirmation: current 1d volume > 1.5x 20-period SMA (volume spike)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (favorable for mean reversion)
        ranging_market = chop[i] > 61.8
        
        # Williams %R signals
        oversold = williams_r[i] < -80      # Oversold condition
        overbought = williams_r[i] > -20    # Overbought condition
        exit_long = williams_r[i] > -50     # Exit long when crosses above -50
        exit_short = williams_r[i] < -50    # Exit short when crosses below -50
        
        # Entry conditions: Williams %R extreme with volume and regime confirmation
        long_entry = oversold and vol_confirm and ranging_market
        short_entry = overbought and vol_confirm and ranging_market
        
        # Exit conditions: Williams %R crosses back through -50
        long_exit = exit_long
        short_exit = exit_short
        
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