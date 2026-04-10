#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level + 1d volume > 2.0x 20-period volume SMA + Chop(14) > 61.8 (range regime)
# - Short when price breaks below Camarilla L3 level + 1d volume > 2.0x 20-period volume SMA + Chop(14) > 61.8
# - Exit: price returns to Camarilla pivot point (mean reversion within pivot structure)
# - Position sizing: 0.25 discrete level
# - Camarilla pivots provide intraday support/resistance levels from prior day
# - Volume confirms institutional participation, chop filter ensures we trade in ranging markets
# - Works in bull/bear: mean reversion at pivot point works in all regimes, chop filter avoids strong trends
# - 4h timeframe targets 19-50 trades/year with strict entry conditions to minimize fee drag

name = "4h_1d_camarilla_volume_chop_v1"
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
    
    # Calculate 1d OHLC for Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 1d: based on previous day's OHLC
    # Camarilla formulas: Pivot = (H+L+C)/3, Range = H-L
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H3 = Pivot + 1.1 * Range / 2, L3 = Pivot - 1.1 * Range / 2
    camarilla_h3_1d = pivot_1d + 1.1 * range_1d / 2
    camarilla_l3_1d = pivot_1d - 1.1 * range_1d / 2
    camarilla_pivot_1d = pivot_1d
    
    # Align Camarilla levels to 4h timeframe (using completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Calculate 1d volume SMA(20) for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume for volume spike confirmation
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)
        
        # Volume confirmation: current 1d volume > 2.0x 20-period SMA (volume spike)
        vol_confirm = vol_1d_current[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Regime filter: Chop > 61.8 indicates ranging market (favorable for mean reversion at pivot)
        ranging_market = chop[i] > 61.8
        
        # Camarilla breakout signals
        breakout_up = close[i] > camarilla_h3_aligned[i]   # Price breaks above H3
        breakout_down = close[i] < camarilla_l3_aligned[i] # Price breaks below L3
        return_to_pivot = abs(close[i] - camarilla_pivot_aligned[i]) < (camarilla_h3_aligned[i] - camarilla_l3_aligned[i]) * 0.3  # Within 30% of pivot
        
        # Entry conditions: Camarilla breakout with volume and regime confirmation
        long_entry = breakout_up and vol_confirm and ranging_market
        short_entry = breakout_down and vol_confirm and ranging_market
        
        # Exit conditions: price returns to Camarilla pivot (mean reversion)
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