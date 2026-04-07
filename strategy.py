#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Camarilla pivot levels from daily + volume spike + choppiness regime filter
# Hypothesis: Camarilla levels act as strong support/resistance; price reversals at these levels
# with volume confirmation and in low-chop (trending) environments yield high-probability trades.
# Works in bull via bounces at L3/L4, in bear via rejections at H3/H4. Choppiness filter avoids whipsaws in ranges.
# Target: 15-30 trades/year to minimize fee drag.
name = "12h_camarilla_daily_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Formula: H4 = C + 1.1*(H-L)/2, H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4, L4 = C - 1.1*(H-L)/2
    # where C, H, L are close, high, low of previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate levels
    H4 = prev_close + 1.1 * (prev_high - prev_low) / 2
    H3 = prev_close + 1.1 * (prev_high - prev_low) / 4
    L3 = prev_close - 1.1 * (prev_high - prev_low) / 4
    L4 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Calculate volume spike: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    # Calculate Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_vals = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    # Sum of ATR over 14 periods
    sum_atr = pd.Series(atr_vals).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_atr / range_hl) / np.log10(14)
    # Handle invalid values
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50, chop)
    
    # Regime filter: CHOP < 40 = trending (favor trend continuation), CHOP > 60 = ranging (favor mean reversion)
    # For this strategy, we prefer trending markets (CHOP < 40) for breakouts
    trending_regime = chop < 40
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(H4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(L4_aligned[i]) or
            np.isnan(vol_spike[i]) or np.isnan(trending_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (mean reversion) or chop becomes too high
            if close[i] < L3_aligned[i] or trending_regime[i] == False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price closes above H3 (mean reversion) or chop becomes too high
            if close[i] > H3_aligned[i] or trending_regime[i] == False:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price closes above H3 with volume spike and in trending regime
            if close[i] > H3_aligned[i] and vol_spike[i] and trending_regime[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below L3 with volume spike and in trending regime
            elif close[i] < L3_aligned[i] and vol_spike[i] and trending_regime[i]:
                position = -1
                signals[i] = -0.25
    
    return signals