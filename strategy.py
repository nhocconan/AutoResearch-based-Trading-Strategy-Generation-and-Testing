#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly pivot points with 1d volume regime filter
# Weekly pivot points (PP, R1/S1, R2/S2, R3/S3) from 1w provide major support/resistance levels
# Volume regime filter (1d volume > 1.2x 20-period average) ensures we only trade during institutional participation
# Fade at R3/S3 (mean reversion) and breakout at R4/S4 (continuation) with 6h price action confirmation
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) with discrete position sizing

name = "6h_1w_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (using previous week's OHLC)
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L, S1 = 2*PP - H
    # R2 = PP + (H - L), S2 = PP - (H - L)
    # R3 = H + 2*(PP - L), S3 = L - 2*(H - PP)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)  # Extension levels
    
    # Shift by 1 to use previous week's data (no look-ahead)
    pp_1w = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    r1_1w = 2 * pp_1w - np.roll(low_1w, 1)
    s1_1w = 2 * pp_1w - np.roll(high_1w, 1)
    r2_1w = pp_1w + (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    s2_1w = pp_1w - (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    r3_1w = np.roll(high_1w, 1) + 2 * (pp_1w - np.roll(low_1w, 1))
    s3_1w = np.roll(low_1w, 1) - 2 * (np.roll(high_1w, 1) - pp_1w)
    r4_1w = r3_1w + (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    s4_1w = s3_1w - (np.roll(high_1w, 1) - np.roll(low_1w, 1))
    
    # Handle first week (no previous data)
    pp_1w[0] = np.nan
    r1_1w[0] = np.nan
    s1_1w[0] = np.nan
    r2_1w[0] = np.nan
    s2_1w[0] = np.nan
    r3_1w[0] = np.nan
    s3_1w[0] = np.nan
    r4_1w[0] = np.nan
    s4_1w[0] = np.nan
    
    # Align weekly pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Load 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # Calculate 20-period average volume on 1d
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Align to 6h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-compute 6h volume average for entry timing
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(vol_ma_20_6h[i]) or vol_ma_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume regime filter: 1d volume > 1.2x 20-period average (institutional participation)
        volume_regime = volume_1d[i // 16] > 1.2 * vol_ma_20_1d[i // 16] if i // 16 < len(volume_1d) else False
        
        # Additional 6h volume confirmation for entry timing
        volume_confirmed = volume[i] > 1.0 * vol_ma_20_6h[i]  # At least average volume
        
        if not (volume_regime and volume_confirmed):
            signals[i] = 0.0
            continue
        
        # Discrete position sizing: 0.25 for all trades
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: 
            # 1. Mean reversion target at weekly PP
            # 2. Stop loss if price breaks below S3 (failed mean reversion)
            if close[i] <= pp_aligned[i] or close[i] < s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Mean reversion target at weekly PP
            # 2. Stop loss if price breaks above R3 (failed mean reversion)
            if close[i] >= pp_aligned[i] or close[i] > r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic:
            # Long: fade at S3 (price < S3 and closing back above S3) OR breakout above R4
            # Short: fade at R3 (price > R3 and closing back below R3) OR breakdown below S4
            
            # Fade at S3/S1 (mean reversion from extreme levels)
            long_fade_s3 = (close[i] > s3_aligned[i]) and (np.roll(close, 1)[i] <= s3_aligned[i])
            long_fade_s1 = (close[i] > s1_aligned[i]) and (np.roll(close, 1)[i] <= s1_aligned[i]) and (close[i] < pp_aligned[i])
            
            # Breakout at R4 (continuation)
            long_breakout = close[i] > r4_aligned[i]
            
            # Fade at R3/R1 (mean reversion from extreme levels)
            short_fade_r3 = (close[i] < r3_aligned[i]) and (np.roll(close, 1)[i] >= r3_aligned[i])
            short_fade_r1 = (close[i] < r1_aligned[i]) and (np.roll(close, 1)[i] >= r1_aligned[i]) and (close[i] > pp_aligned[i])
            
            # Breakdown at S4 (continuation)
            short_breakdown = close[i] < s4_aligned[i]
            
            if (long_fade_s3 or long_fade_s1 or long_breakout) and volume_confirmed:
                position = 1
                signals[i] = position_size
            elif (short_fade_r3 or short_fade_r1 or short_breakdown) and volume_confirmed:
                position = -1
                signals[i] = -position_size
    
    return signals