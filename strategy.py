#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot levels from 1d + volume confirmation + choppiness regime filter
# Camarilla pivots identify key support/resistance levels where price often reverses or breaks out
# Volume confirmation ensures breakouts have conviction
# Choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) adapts strategy to market conditions
# In ranging markets: mean revert at Camarilla H3/L3 levels
# In trending markets: breakout through H4/L4 levels with volume
# Works in bull/bear: regime filter prevents false signals in wrong market conditions
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "12h_1d_camarilla_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla pivots and choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h6 = np.full(n, np.nan)
    camarilla_l6 = np.full(n, np.nan)
    
    for i in range(n):
        if i < 1:
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
            camarilla_h3[i] = np.nan
            camarilla_l3[i] = np.nan
            camarilla_h6[i] = np.nan
            camarilla_l6[i] = np.nan
        else:
            # Previous day's OHLC
            phigh = df_1d['high'].values[i-1]
            plow = df_1d['low'].values[i-1]
            pclose = df_1d['close'].values[i-1]
            
            pivot = (phigh + plow + pclose) / 3
            range_val = phigh - plow
            
            camarilla_h4[i] = pivot + range_val * 1.1 / 2
            camarilla_l4[i] = pivot - range_val * 1.1 / 2
            camarilla_h3[i] = pivot + range_val * 1.1 / 4
            camarilla_l3[i] = pivot - range_val * 1.1 / 4
            camarilla_h6[i] = pivot + range_val * 1.1 / 6
            camarilla_l6[i] = pivot - range_val * 1.1 / 6
    
    # Align Camarilla levels to 12h timeframe (already aligned by get_htf_data + align_htf_to_ltf logic in helper)
    # Actually, get_htf_data gives us 1d data, we need to align our calculated values
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h6)
    camarilla_l6_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l6)
    
    # Calculate 1d choppiness index for regime filter
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            chop[i] = np.nan
        else:
            # True range calculation
            tr1 = df_1d['high'].values[i-14:i] - df_1d['low'].values[i-14:i]
            tr2 = np.abs(df_1d['high'].values[i-14:i] - df_1d['close'].values[i-15:i-1])
            tr3 = np.abs(df_1d['low'].values[i-14:i] - df_1d['close'].values[i-15:i-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            atr = np.mean(tr)
            max_high = np.max(df_1d['high'].values[i-14:i])
            min_low = np.min(df_1d['low'].values[i-14:i])
            
            if atr == 0:
                chop[i] = 50
            else:
                chop[i] = 100 * np.log10(atr / (max_high - min_low)) / np.log10(14)
    
    # Align choppiness to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Regime filter: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (breakout)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        if position == 1:  # Long position
            # Exit logic
            if is_ranging:
                # In ranging market: exit at L3 (profit target) or H4 (stop loss)
                if close[i] <= camarilla_l3_aligned[i] or close[i] >= camarilla_h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # trending market
                # In trending market: exit at L4 (stop loss) or H6 (profit target)
                if close[i] <= camarilla_l4_aligned[i] or close[i] >= camarilla_h6_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit logic
            if is_ranging:
                # In ranging market: exit at H3 (profit target) or L4 (stop loss)
                if close[i] >= camarilla_h3_aligned[i] or close[i] <= camarilla_l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # trending market
                # In trending market: exit at H4 (stop loss) or L6 (profit target)
                if close[i] >= camarilla_h4_aligned[i] or close[i] <= camarilla_l6_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic
            if volume_confirmed:
                if is_ranging:
                    # In ranging market: mean revert at H3/L3 levels
                    if close[i] <= camarilla_l3_aligned[i] and close[i] > camarilla_l4_aligned[i]:
                        # Long near L3 with stop at L4
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= camarilla_h3_aligned[i] and close[i] < camarilla_h4_aligned[i]:
                        # Short near H3 with stop at H4
                        position = -1
                        signals[i] = -0.25
                else:  # is_trending
                    # In trending market: breakout through H4/L4 levels
                    if close[i] > camarilla_h4_aligned[i] and close[i] > camarilla_h6_aligned[i]:
                        # Strong breakout above H4
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < camarilla_l4_aligned[i] and close[i] < camarilla_l6_aligned[i]:
                        # Strong breakdown below L4
                        position = -1
                        signals[i] = -0.25
    
    return signals