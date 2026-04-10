#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1d volume spike filter and 1d choppiness regime filter
# - Entry: Long when price breaks above Camarilla H3 + 1d volume > 2.0x 20-period average + 1d Choppiness Index > 61.8 (range regime)
#          Short when price breaks below Camarilla L3 + 1d volume > 2.0x 20-period average + 1d Choppiness Index > 61.8 (range regime)
# - Exit: Close-based reversal - exit long when price < Camarilla H3 level, exit short when price > Camarilla L3 level
# - Position sizing: 0.25 (discrete levels to minimize fee churn)
# - Uses Camarilla pivot levels from daily data for structure, daily volume spike for confirmation of participation,
#   and daily Choppiness Index to filter for range-bound markets where mean reversion at pivots works best
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within HARD MAX: 200 total
# - Choppiness Index > 61.8 indicates ranging market (ideal for Camarilla mean reversion), < 38.2 indicates trending
# - Volume spike ensures genuine participation at breakout, reducing false signals
# - 12h timeframe provides sufficient signal quality while controlling trade frequency

name = "12h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLC
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Pre-compute 1d data for indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_h3 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 4
    camarilla_l3 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 4
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Choppiness Index (14-period)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over period
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over period
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(tr) / (hh - ll)) / log10(period)
    # Avoid division by zero
    range_hl = hh_1d - ll_1d
    choppiness = np.zeros_like(sum_tr)
    mask = (range_hl > 0) & (~np.isnan(sum_tr)) & (~np.isnan(range_hl))
    choppiness[mask] = 100 * np.log10(sum_tr[mask] / range_hl[mask]) / np.log10(14)
    
    # Align all HTF data to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(choppiness_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Get current 1d volume for confirmation
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirmation = volume_1d_aligned[i] > 2.0 * volume_ma_aligned[i]
        
        # Choppiness filter: > 61.8 indicates ranging market (good for mean reversion at pivots)
        chop_filter = choppiness_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla H3 + volume confirmation + ranging market
            if (close_price > camarilla_h3_aligned[i] and 
                volume_confirmation and 
                chop_filter):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Camarilla L3 + volume confirmation + ranging market
            elif (close_price < camarilla_l3_aligned[i] and 
                  volume_confirmation and 
                  chop_filter):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit long when price < Camarilla H3 level
            # Exit short when price > Camarilla L3 level
            if position == 1:
                if close_price < camarilla_h3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close_price > camarilla_l3_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals