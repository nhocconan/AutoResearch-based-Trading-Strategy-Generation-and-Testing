#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and chop regime filter
# - Long when price breaks above Camarilla H3 level AND 1d volume > 1.5x 20-period volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below Camarilla L3 level AND 1d volume > 1.5x 20-period volume SMA AND chop > 61.8
# - Exit: price crosses Camarilla pivot point (midline)
# - Uses 4h for price action (Camarilla levels), 1d for volume confirmation, 4h for chop filter
# - Camarilla pivots identify key support/resistance; volume confirms institutional interest; chop filter avoids strong trends
# - Target: 20-30 trades/year to minimize fee drag while capturing high-probability mean reversion in ranging markets

name = "4h_1d_camarilla_volspike_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Calculate 1d volume SMA for confirmation
    vol_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 4h Camarilla levels (based on previous day's OHLC)
    # Camarilla levels require daily OHLC, so we need to get the previous day's values
    # We'll use the 1d data to calculate Camarilla levels for each 4h bar
    # For each 4h bar, we use the previous 1d bar's OHLC
    
    # Extract 1d OHLC
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Pivot = (high + low + close) / 3
    camarilla_h3_1d = close_1d + 1.1 * (high_1d - low_1d) / 4.0
    camarilla_l3_1d = close_1d - 1.1 * (high_1d - low_1d) / 4.0
    camarilla_pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot_1d)
    
    # Pre-compute 4h Chopiness Index (14-period) for regime filter
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop = np.where((highest_high_14 - lowest_low_14) == 0, 50, chop)  # avoid division by zero
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = vol_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Chop filter: chop > 61.8 indicates ranging market (good for breakout mean reversion)
        chop_filter = chop[i] > 61.8
        
        # Only trade when both volume confirmation and chop filter are present
        if vol_confirm and chop_filter:
            # Long: price breaks above Camarilla H3
            if close[i] > camarilla_h3_aligned[i]:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: price breaks below Camarilla L3
            elif close[i] < camarilla_l3_aligned[i]:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            
            # Exit conditions: price crosses Camarilla pivot point
            if position == 1 and close[i] < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
        else:
            # No trade: exit any position if conditions not met
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals