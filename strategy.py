#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Camarilla pivot levels with volume confirmation and choppiness regime filter
# Camarilla pivots from 12h provide intraday support/resistance that work in both bull and bear markets
# Volume confirmation (current 4h volume > 1.3x 20-period average) filters false breakouts
# Choppiness regime filter (CHOP(14) > 61.8 = range, < 38.2 = trending) adapts to market conditions
# In ranging markets (CHOP > 61.8): mean reversion at Camarilla H3/L3 levels
# In trending markets (CHOP < 38.2): breakout continuation at H4/L4 levels
# Position sizing: 0.25 for consistency
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_12h_camarilla_volume_chop_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 15:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla pivot levels (based on previous 12h bar)
    # Camarilla formula: 
    # H4 = close + 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    # L4 = close - 1.1*(high-low)/2
    # Using previous 12h bar to avoid look-ahead
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    prev_high[0] = high_12h[0]  # First period
    prev_low[0] = low_12h[0]
    prev_close[0] = close_12h[0]
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 4.0
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 4.0
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h4)
    h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3)
    l4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l4)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14-period) for 4h
    # CHOP = 100 * log10(sum(ATR(1)) / (n * ATR(n))) / log10(n)
    # Where ATR(1) = true range, ATR(n) = n-period ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_1 = tr  # True range
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of true range over 14 periods
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Choppiness Index
    chop = 100 * np.log10(sum_tr_14 / (14 * atr_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((atr_14 > 0) & (sum_tr_14 > 0), chop, 50.0)  # Default to 50 (neutral)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(chop[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.3x average 4h volume
        volume_confirmed = volume[i] > 1.3 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Regime filter based on Choppiness Index
        # CHOP > 61.8 = ranging market (mean revert)
        # CHOP < 38.2 = trending market (breakout)
        # 38.2 <= CHOP <= 61.8 = transition (no trade)
        if chop[i] > 61.8:
            # Ranging market: mean reversion at H3/L3
            if position == 1:  # Long position
                # Exit on reversion to midpoint or breakdown below L3
                if close[i] < (h3_aligned[i] + l3_aligned[i]) / 2.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit on reversion to midpoint or breakout above H3
                if close[i] > (h3_aligned[i] + l3_aligned[i]) / 2.0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat
                # Mean reversion entries
                if close[i] <= l3_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] >= h3_aligned[i]:
                    position = -1
                    signals[i] = -0.25
        elif chop[i] < 38.2:
            # Trending market: breakout continuation at H4/L4
            if position == 1:  # Long position
                # Exit on breakdown below H4
                if close[i] < h4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:  # Short position
                # Exit on breakout above L4
                if close[i] > l4_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Flat
                # Breakout entries
                if close[i] > h4_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < l4_aligned[i]:
                    position = -1
                    signals[i] = -0.25
        else:
            # Transition regime: no trading
            signals[i] = 0.0
    
    return signals