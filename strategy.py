#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot levels + volume spike + choppiness regime filter
# Camarilla levels provide intraday support/resistance with high probability reversal zones
# Volume spike confirms institutional participation at key levels
# Choppiness filter avoids whipsaws in strong trends (use range-bound entries)
# Works in bull/bear: mean reversion at pivots effective in ranging markets
# Target: 50-100 total trades over 4 years (12-25/year) with discrete sizing

name = "1d_camarilla_pivot_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data (same as primary) for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    camarilla_h4 = np.full(n, np.nan)  # Resistance 4
    camarilla_h3 = np.full(n, np.nan)  # Resistance 3
    camarilla_h2 = np.full(n, np.nan)  # Resistance 2
    camarilla_h1 = np.full(n, np.nan)  # Resistance 1
    camarilla_l1 = np.full(n, np.nan)  # Support 1
    camarilla_l2 = np.full(n, np.nan)  # Support 2
    camarilla_l3 = np.full(n, np.nan)  # Support 3
    camarilla_l4 = np.full(n, np.nan)  # Support 4
    
    for i in range(1, n):
        # Use previous day's OHLC (bar i-1)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_val = prev_high - prev_low
        
        # Camarilla levels
        camarilla_h4[i] = prev_close + range_val * 1.5 / 2
        camarilla_h3[i] = prev_close + range_val * 1.25 / 2
        camarilla_h2[i] = prev_close + range_val * 1.1666 / 2
        camarilla_h1[i] = prev_close + range_val * 1.0833 / 2
        camarilla_l1[i] = prev_close - range_val * 1.0833 / 2
        camarilla_l2[i] = prev_close - range_val * 1.1666 / 2
        camarilla_l3[i] = prev_close - range_val * 1.25 / 2
        camarilla_l4[i] = prev_close - range_val * 1.5 / 2
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    # Calculate choppiness index (14-period) for regime filter
    chop = np.full(n, np.nan)
    for i in range(n):
        if i < 14:
            chop[i] = np.nan
        else:
            # True range calculation
            tr1 = high[i-13:i+1] - low[i-13:i+1]
            tr2 = np.abs(high[i-13:i+1] - np.roll(close[i-13:i+1], 1))
            tr3 = np.abs(low[i-13:i+1] - np.roll(close[i-13:i+1], 1))
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            # Set first TR to high-low (no previous close)
            tr[0] = high[i-13] - low[i-13]
            
            atr_sum = np.sum(tr)
            max_high = np.max(high[i-13:i+1])
            min_low = np.min(low[i-13:i+1])
            
            if atr_sum > 0 and max_high > min_low:
                chop[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
            else:
                chop[i] = 50.0  # neutral
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        
        # Choppiness regime: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: price < Camarilla L3 (strong support broken) OR chop < 38.2 (trending)
            if close[i] < camarilla_l3[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Camarilla H3 (strong resistance broken) OR chop < 38.2 (trending)
            if close[i] > camarilla_h3[i] or chop[i] < 38.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Camarilla touch in ranging market
            if volume_confirmed and ranging_market:
                # Long entry: price touches Camarilla L4 (extreme support) with rejection
                if low[i] <= camarilla_l4[i] and close[i] > camarilla_l4[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price touches Camarilla H4 (extreme resistance) with rejection
                elif high[i] >= camarilla_h4[i] and close[i] < camarilla_h4[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals