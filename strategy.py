#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level + 12h volume > 2.0x 20-period volume SMA + Chop(14) > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level + 12h volume > 2.0x 20-period volume SMA + Chop(14) > 61.8
# - Exit: price returns to Camarilla pivot point (mean reversion within range)
# - Position sizing: 0.25 discrete level
# - Camarilla levels derived from prior 12h bar provide structure in ranging markets
# - Volume confirms institutional participation, chop filter ensures we trade in ranging markets
# - Works in bull/bear: mean reversion at pivot levels works in all regimes, chop filter avoids strong trends
# - 4h timeframe targets 20-50 trades/year with strict entry conditions to minimize fee drag

name = "4h_12h_camarilla_vol_chop_v1"
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
    
    # Calculate 12h Camarilla pivot levels (based on prior 12h bar)
    # Camarilla formula: H4 = close + 1.1*(high-low)*1.1/2, H3 = close + 1.1*(high-low)*1.1/4
    # L3 = close - 1.1*(high-low)*1.1/4, L4 = close - 1.1*(high-low)*1.1/2
    # Pivot = (high + low + close) / 3
    hl_range_12h = df_12h['high'].values - df_12h['low'].values
    camarilla_pivot = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3
    camarilla_h3 = camarilla_pivot + 1.1 * hl_range_12h * 1.1 / 4
    camarilla_l3 = camarilla_pivot - 1.1 * hl_range_12h * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (using completed 12h bar)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    
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
        if (np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h volume for volume spike confirmation
        vol_12h_current = align_htf_to_ltf(prices, df_12h, df_12h['volume'].values)
        
        # Volume confirmation: current 12h volume > 2.0x 20-period SMA (volume spike)
        vol_confirm = vol_12h_current[i] > 2.0 * volume_sma_20_12h_aligned[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (favorable for mean reversion at pivot levels)
        ranging_market = chop[i] > 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]   # Price breaks above H3 level
        breakout_down = close[i] < camarilla_l3_aligned[i] # Price breaks below L3 level
        return_to_pivot = abs(close[i] - camarilla_pivot_aligned[i]) < (camarilla_h3_aligned[i] - camarilla_l3_aligned[i]) * 0.15  # Within 15% of pivot
        
        # Entry conditions: Camarilla breakout with volume and regime confirmation
        long_entry = breakout_up and vol_confirm and ranging_market
        short_entry = breakout_down and vol_confirm and ranging_market
        
        # Exit conditions: price returns to Camarilla pivot point (mean reversion)
        long_exit = return_to_pivot  # Exit long when price returns to pivot
        short_exit = return_to_pivot  # Exit short when price returns to pivot
        
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