#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Uses actual 1d Camarilla calculations (proven ETH edge) with 12h execution
# Volume confirmation ensures breakout authenticity; chop filter avoids whipsaws
# Discrete sizing 0.25 limits fee drag; works in bull/bear via regime adaptation
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1d_camarilla_volume_chop_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h2 = np.full(n, np.nan)
    camarilla_l2 = np.full(n, np.nan)
    camarilla_h1 = np.full(n, np.nan)
    camarilla_l1 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:
            # Need previous day's data
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h2[i] = np.nan
            camarilla_l2[i] = np.nan
            camarilla_h1[i] = np.nan
            camarilla_l1[i] = np.nan
        else:
            # Use previous 1d bar's OHLC (yesterday's close)
            prev_high = df_1d['high'].values[i-1] if i-1 < len(df_1d) else df_1d['high'].values[-1]
            prev_low = df_1d['low'].values[i-1] if i-1 < len(df_1d) else df_1d['low'].values[-1]
            prev_close = df_1d['close'].values[i-1] if i-1 < len(df_1d) else df_1d['close'].values[-1]
            
            # Camarilla calculations
            range_val = prev_high - prev_low
            camarilla_h4[i] = prev_close + range_val * 1.1 / 2
            camarilla_l4[i] = prev_close - range_val * 1.1 / 2
            camarilla_h3[i] = prev_close + range_val * 1.1 / 4
            camarilla_l3[i] = prev_close - range_val * 1.1 / 4
            camarilla_h2[i] = prev_close + range_val * 1.1 / 6
            camarilla_l2[i] = prev_close - range_val * 1.1 / 6
            camarilla_h1[i] = prev_close + range_val * 1.1 / 12
            camarilla_l1[i] = prev_close - range_val * 1.1 / 12
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar close)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate Choppiness Index (14-period) for regime filter
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            chop[i] = np.nan
        else:
            # True Range calculation
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Sum of TR over 14 periods
            sum_tr = 0
            for j in range(14):
                idx = i - j
                if idx >= 0:
                    tr1_j = high[idx] - low[idx]
                    tr2_j = abs(high[idx] - close[idx-1]) if idx > 0 else 0
                    tr3_j = abs(low[idx] - close[idx-1]) if idx > 0 else 0
                    tr_j = np.maximum(tr1_j, np.maximum(tr2_j, tr3_j))
                    sum_tr += tr_j
            
            # Max high and min low over 14 periods
            max_high = np.max(high[i-13:i+1])
            min_low = np.min(low[i-13:i+1])
            
            if max_high > min_low and sum_tr > 0:
                chop[i] = 100 * np.log10(sum_tr / (max_high - min_low)) / np.log10(14)
            else:
                chop[i] = 50.0  # neutral
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Chop regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        # In ranging markets: mean revert at H3/L3 levels
        # In trending markets: breakout at H4/L4 levels
        if chop[i] > 61.8:  # Ranging market - mean revert
            if position == 1:  # Long position
                # Exit: price < L3 (mean reversion target) OR chop shifts to trending
                if close[i] < l3_aligned[i] or chop[i] < 38.2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: price > H3 (mean reversion target) OR chop shifts to trending
                if close[i] > h3_aligned[i] or chop[i] < 38.2:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat
                # Entry logic for ranging market
                if volume_confirmed:
                    # Long entry: price < L3 AND showing bounce (close > open)
                    if close[i] < l3_aligned[i] and close[i] > prices['open'].iloc[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short entry: price > H3 AND showing rejection (close < open)
                    elif close[i] > h3_aligned[i] and close[i] < prices['open'].iloc[i]:
                        position = -1
                        signals[i] = -0.25
        else:  # Trending market (CHOP < 61.8)
            if position == 1:  # Long position
                # Exit: price < L4 (breakdown) OR chop shifts to strong ranging
                if close[i] < l4_aligned[i] or chop[i] > 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit: price > H4 (breakout) OR chop shifts to strong ranging
                if close[i] > h4_aligned[i] or chop[i] > 61.8:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat
                # Entry logic for trending market
                if volume_confirmed:
                    # Long entry: price > H4 (breakout) AND close > open (confirmation)
                    if close[i] > h4_aligned[i] and close[i] > prices['open'].iloc[i]:
                        position = 1
                        signals[i] = 0.25
                    # Short entry: price < L4 (breakdown) AND close < open (confirmation)
                    elif close[i] < l4_aligned[i] and close[i] < prices['open'].iloc[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals