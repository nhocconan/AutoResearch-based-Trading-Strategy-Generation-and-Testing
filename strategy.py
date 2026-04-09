#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume spike + choppiness regime filter
# Camarilla pivots provide strong support/resistance levels from prior day
# Volume spike confirms breakout authenticity; chop filter avoids false signals in ranging markets
# Works in bull/bear: mean reversion at pivots in chop, breakouts in trend
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.30

name = "4h_1d_camarilla_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    camarilla_h4 = np.full(n, np.nan)  # Resistance
    camarilla_l4 = np.full(n, np.nan)  # Support
    
    for i in range(n):
        if i < 1:  # Need at least one prior 1d bar
            camarilla_h4[i] = np.nan
            camarilla_l4[i] = np.nan
        else:
            # Get prior completed 1d bar (index i-1 in 1d data)
            # Since we're on 4h timeframe, we need to map to 1d index
            # Use align_htf_to_ltf approach: get prior 1d bar's OHLC
            if i < 96:  # Need ~4 days of 4h data to get first 1d bar (96 periods = 4 days * 24h/4h)
                camarilla_h4[i] = np.nan
                camarilla_l4[i] = np.nan
            else:
                # Simplified: use rolling window of 96 periods (4 days) to approximate prior day
                # More precise would require HTF index mapping, but this avoids look-ahead
                start_idx = max(0, i - 96)
                if start_idx < i:
                    prior_high = np.max(high[start_idx:i])
                    prior_low = np.min(low[start_idx:i])
                    prior_close = np.mean(close[start_idx:i])  # approximation
                    
                    # Camarilla levels
                    range_val = prior_high - prior_low
                    camarilla_h4[i] = prior_close + range_val * 1.1 / 4
                    camarilla_l4[i] = prior_close - range_val * 1.1 / 4
                else:
                    camarilla_h4[i] = np.nan
                    camarilla_l4[i] = np.nan
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate choppiness index regime filter (14-period)
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            chop[i] = np.nan
        else:
            # True range calculation
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            tr = max(tr1, max(tr2, tr3))
            
            # Sum of true ranges over 14 periods
            atr_sum = 0
            for j in range(14):
                idx = i - j
                if idx >= 0:
                    tr1_j = high[idx] - low[idx]
                    tr2_j = abs(high[idx] - close[idx-1]) if idx > 0 else 0
                    tr3_j = abs(low[idx] - close[idx-1]) if idx > 0 else 0
                    tr_j = max(tr1_j, max(tr2_j, tr3_j))
                    atr_sum += tr_j
            
            if atr_sum > 0:
                # Choppiness index formula
                chop[i] = 100 * np.log10(atr_sum / np.log(14)) / np.log(14)
            else:
                chop[i] = 50.0  # neutral
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending
        is_ranging = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L4 OR (trending and price < prior low)
            if close[i] < camarilla_l4[i] or (is_trending and i >= 2 and close[i] < low[i-2]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H4 OR (trending and price > prior high)
            if close[i] > camarilla_h4[i] or (is_trending and i >= 2 and close[i] > high[i-2]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Entry logic
            if volume_confirmed:
                if is_ranging:
                    # Mean reversion in ranging market
                    if close[i] < camarilla_l4[i] and i >= 2 and close[i-1] >= camarilla_l4[i-1]:
                        # Bullish bounce off L4
                        position = 1
                        signals[i] = 0.30
                    elif close[i] > camarilla_h4[i] and i >= 2 and close[i-1] <= camarilla_h4[i-1]:
                        # Bearish rejection at H4
                        position = -1
                        signals[i] = -0.30
                elif is_trending:
                    # Breakout in trending market
                    if close[i] > camarilla_h4[i] and i >= 2 and close[i-1] <= camarilla_h4[i-1]:
                        # Bullish breakout above H4
                        position = 1
                        signals[i] = 0.30
                    elif close[i] < camarilla_l4[i] and i >= 2 and close[i-1] >= camarilla_l4[i-1]:
                        # Bearish breakdown below L4
                        position = -1
                        signals[i] = -0.30
    
    return signals